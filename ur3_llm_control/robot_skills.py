import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import yaml
import json
import os
import math
import time
import numpy as np
import threading

from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, RobotState
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.msg import (
    Constraints, 
    JointConstraint,
    PlanningScene,
    CollisionObject,
    AttachedCollisionObject,
    MoveItErrorCodes
)
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener
from linkattacher_msgs.srv import AttachLink, DetachLink
from scipy.spatial.transform import Rotation



def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw

class RobotSkills:
    def __init__(self, node: Node, config_path: str = None):
        self.node = node
        self.logger = node.get_logger()
        
        if config_path is None:
            config_path = '/home/gdon/ur_ws/src/ur3_llm_control/config/scene.yaml'
        
        self.config = self._load_config(config_path)
        self.m = self.config.get('motion', {})
        self.g = self.config.get('gripper', {})
        self.objects = self.config.get('objects', {})
        self.zones = self.config.get('zones', {})
        self.staging_positions = self.config.get('staging_positions', {})

        self.current_grasp_yaw = 0.0
        self.cube_size = float(self.g.get('cube_size', 0.04))
        self.open_val = float(self.g.get('open_pos', 0.045))
        self.close_val = float(self.g.get('close_pos', 0.018))
        self.approach_z = float(self.m.get('approach_z', 0.22))
        self.grasp_z = float(self.m.get('grasp_z', 0.130))
        self.place_z = float(self.m.get('place_z', 0.132))

        self._ik_client = self.node.create_client(GetPositionIK, 'compute_ik')
        self._move_group_client = ActionClient(self.node, MoveGroup, 'move_action')
        self._execute_traj_client = ActionClient(self.node, ExecuteTrajectory, 'execute_trajectory')
        self._cartesian_client = self.node.create_client(GetCartesianPath, 'compute_cartesian_path')
        self._scene_pub = self.node.create_publisher(PlanningScene, '/planning_scene', 10)

        self._gripper_pub = self.node.create_publisher(
            Float64MultiArray, 
            '/gripper_controller/commands', 
            10
        )
        self._left_finger_pos = self.open_val
        self._right_finger_pos = self.open_val
        self._current_gap = self._left_finger_pos + self._right_finger_pos
        self._joint_sub = self.node.create_subscription(
            JointState, 
            '/joint_states', 
            self._joint_state_callback, 
            10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self.attached_object = None

        self._attach_client = self.node.create_client(AttachLink, '/ATTACHLINK')
        self._detach_client = self.node.create_client(DetachLink, '/DETACHLINK')

        self.logger.info("Đang chờ MoveGroup Action Server...")
        self._move_group_client.wait_for_server(timeout_sec=5.0)
        self._init_planning_scene()

    def spin_sleep(self, duration: float):
        """Ngủ nhưng liên tục spin node để nhận và xử lý các callback ROS (JointState, TF, Camera)"""
        start = time.time()
        while (time.time() - start) < duration and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def _load_config(self, path):
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.error(f"Lỗi đọc config: {e}")
            return {}

    def _joint_state_callback(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name == 'left_finger_joint':
                self._left_finger_pos = float(pos)
            elif name == 'right_finger_joint':
                self._right_finger_pos = float(pos)
        self._current_gap = self._left_finger_pos + self._right_finger_pos

    def update_detected_objects(self, detected_objects: dict):
        for obj_name, coords in detected_objects.items():
            if obj_name == self.attached_object:
                continue
            x = float(coords.get('x', 0))
            y = float(coords.get('y', 0))
            # Chỉ cập nhật nếu tọa độ hợp lệ (nằm trong phạm vi mặt bàn làm việc)
            if 0.15 <= x <= 0.60 and -0.35 <= y <= 0.35:
                if obj_name not in self.objects:
                    self.objects[obj_name] = {}
                self.objects[obj_name]['x'] = x
                self.objects[obj_name]['y'] = y
                self.objects[obj_name]['z'] = float(coords.get('z', 0.06))

    def get_world_state_description(self) -> str:
        """Định dạng trạng thái môi trường hiện tại dựa trên dữ liệu scene và lịch sử thao tác"""
        zones_status = {}
        for z_name, z_coord in self.zones.items():
            occupant = "empty"
            for obj_name, obj_coord in self.objects.items():
                dist = math.hypot(obj_coord['x'] - z_coord['x'], obj_coord['y'] - z_coord['y'])
                if dist < 0.06:
                    occupant = obj_name
                    break
            zones_status[z_name] = occupant

        staging_status = {}
        for s_name, s_coord in self.staging_positions.items():
            occupant = "empty"
            for obj_name, obj_coord in self.objects.items():
                dist = math.hypot(obj_coord['x'] - s_coord['x'], obj_coord['y'] - s_coord['y'])
                if dist < 0.06:
                    occupant = obj_name
                    break
            staging_status[s_name] = occupant

        state = {
            "objects": {k: {"x": round(v['x'], 3), "y": round(v['y'], 3)} for k, v in self.objects.items()},
            "zones_status": zones_status,
            "staging_status": staging_status
        }
        return json.dumps(state, indent=2)

    # ------------------------------------------------------------ PLANNING SCENE
    def _init_planning_scene(self):
        scene_msg = PlanningScene()
        scene_msg.is_diff = True

        table = CollisionObject()
        table.header.frame_id = "base_link"
        table.id = "table"
        b = SolidPrimitive()
        b.type = SolidPrimitive.BOX
        b.dimensions = [0.46, 0.65, 0.04]
        p = PoseStamped()
        p.header.frame_id = "base_link"
        p.pose.position.x = 0.40
        p.pose.position.y = 0.0
        p.pose.position.z = 0.02
        p.pose.orientation.w = 1.0
        table.primitives.append(b)
        table.primitive_poses.append(p.pose)
        table.operation = CollisionObject.ADD
        scene_msg.world.collision_objects.append(table)

        self._scene_pub.publish(scene_msg)

    def _scene_remove(self, obj_name):
        pass

    def _scene_attach(self, obj_name):
        pass

    def _scene_detach(self, obj_name, x, y):
        pass

    # ------------------------------------------------------------ MOTION PLANNING VỚI IK GIẢI TÍCH
    def _compute_ik_approach(self, x, y, z):
        if not self._ik_client.wait_for_service(timeout_sec=2.0):
            self.logger.error("Dịch vụ compute_ik không khả dụng!")
            return None

        # Ưu tiên -pi/2 để ngón kẹp mở vuông góc với hàng cube (dọc trục X)
        candidate_yaws = [-math.pi / 2.0, 0.0, math.pi / 2.0]
        pan = math.atan2(float(y), float(x))
        candidate_seeds = []
        for s_lift in [-1.4, -1.6, -1.2, -1.8]:
            for s_elb in [1.8, 1.5, 1.2, 2.0]:
                s_w1 = -math.pi / 2.0 - (s_lift + s_elb)
                s_w2 = -math.pi / 2.0
                candidate_seeds.append((s_lift, s_elb, s_w1, s_w2))

        for yaw in candidate_yaws:
            for s_lift, s_elb, s_w1, s_w2 in candidate_seeds:
                req = GetPositionIK.Request()
                req.ik_request.group_name = "ur_manipulator"
                req.ik_request.ik_link_name = "tool0"
                req.ik_request.avoid_collisions = True  # Tránh va chạm tự thân và va chạm cảnh quan

                target_pose = PoseStamped()
                target_pose.header.frame_id = "base_link"
                target_pose.pose.position.x = float(x)
                target_pose.pose.position.y = float(y)
                target_pose.pose.position.z = float(z)

                qx, qy, qz, qw = euler_to_quaternion(math.pi, 0.0, yaw)
                target_pose.pose.orientation.x = qx
                target_pose.pose.orientation.y = qy
                target_pose.pose.orientation.z = qz
                target_pose.pose.orientation.w = qw
                req.ik_request.pose_stamped = target_pose

                seed_state = RobotState()
                seed_state.joint_state.name = [
                    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
                ]
                seed_w3 = yaw - pan
                seed_state.joint_state.position = [pan, s_lift, s_elb, s_w1, s_w2, seed_w3]
                req.ik_request.robot_state = seed_state
                req.ik_request.timeout.sec = 1

                future = self._ik_client.call_async(req)
                rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)
                res = future.result()

                if res and res.error_code.val == MoveItErrorCodes.SUCCESS:
                    sol = res.solution.joint_state
                    joint_names = [
                        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
                    ]
                    q_out = []
                    for name in joint_names:
                        if name in sol.name:
                            idx = sol.name.index(name)
                            q_out.append(sol.position[idx])
                    if len(q_out) == 6:
                        pan_val, lift_val, elb_val, w1_val = q_out[0], q_out[1], q_out[2], q_out[3]
                        diff_pan = (pan_val - pan + math.pi) % (2 * math.pi) - math.pi
                        # Chỉ chấp nhận cấu hình UR3 vươn tới tự nhiên (elbow-up, wrist hướng xuống)
                        if -2.5 < lift_val < -0.7 and 0.5 < elb_val < 2.8 and -2.8 < w1_val < -0.4 and abs(diff_pan) < 1.0:
                            self.current_grasp_yaw = yaw
                            return q_out

        self.logger.warn(f"MoveIt IK không tìm được nghiệm cho điểm ({x:.3f}, {y:.3f}, {z:.3f})")
        return None

    def _send_joint_goal(self, joint_positions, planning_group="ur_manipulator"):
        if not self._move_group_client.server_is_ready():
            return "FAILED"

        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = planning_group
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 0.25
        goal_msg.request.max_acceleration_scaling_factor = 0.25

        c = Constraints()
        joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                       'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        
        for name, pos in zip(joint_names, joint_positions):
            j_constraint = JointConstraint()
            j_constraint.joint_name = name
            j_constraint.position = float(pos)
            j_constraint.tolerance_above = 0.01
            j_constraint.tolerance_below = 0.01
            j_constraint.weight = 1.0
            c.joint_constraints.append(j_constraint)

        goal_msg.request.goal_constraints.append(c)

        for attempt in range(2):
            send_future = self._move_group_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self.node, send_future, timeout_sec=5.0)
            goal_handle = send_future.result()

            if not goal_handle or not goal_handle.accepted:
                self.spin_sleep(0.3)
                continue

            res_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self.node, res_future, timeout_sec=10.0)
            result = res_future.result()

            if result and result.result.error_code.val == MoveItErrorCodes.SUCCESS:
                return "SUCCESS"
            self.spin_sleep(0.3)

        return "FAILED"

    def _move_above_xy(self, x, y, z_height=0.22):
        q = self._compute_ik_approach(x, y, z_height)
        if q is None:
            return "PLANNING_FAILED"
        return self._send_joint_goal(q)

    def move_vertical(self, target_z, planning_group="ur_manipulator") -> str:
        """Hạ xuống hoặc nhấc lên theo đường thẳng đứng Cartesian tuyệt đối[cite: 13]"""
        try:
            t = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
            curr_x = t.transform.translation.x
            curr_y = t.transform.translation.y
        except Exception as e:
            self.logger.error(f"Lỗi TF: {e}")
            return "FAILED"

        if not self._cartesian_client.wait_for_service(timeout_sec=2.0):
            return self._move_above_xy(curr_x, curr_y, target_z)

        qx, qy, qz, qw = euler_to_quaternion(math.pi, 0.0, self.current_grasp_yaw)
        target_pose = Pose()
        target_pose.position.x = curr_x
        target_pose.position.y = curr_y
        target_pose.position.z = float(target_z)
        target_pose.orientation.x = qx
        target_pose.orientation.y = qy
        target_pose.orientation.z = qz
        target_pose.orientation.w = qw

        req = GetCartesianPath.Request()
        req.header.frame_id = "base_link"
        req.header.stamp = self.node.get_clock().now().to_msg()
        req.waypoints = [target_pose]
        req.max_step = 0.003
        req.jump_threshold = 0.0
        req.avoid_collisions = False
        req.group_name = planning_group

        future = self._cartesian_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=3.0)
        res = future.result()

        if not res or res.fraction < 0.8:
            return self._move_above_xy(curr_x, curr_y, target_z)

        exec_goal = ExecuteTrajectory.Goal()
        exec_goal.trajectory = res.solution

        send_future = self._execute_traj_client.send_goal_async(exec_goal)
        rclpy.spin_until_future_complete(self.node, send_future, timeout_sec=4.0)
        goal_handle = send_future.result()

        if not goal_handle or not goal_handle.accepted:
            return self._move_above_xy(curr_x, curr_y, target_z)

        res_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, res_future, timeout_sec=8.0)
        exec_res = res_future.result()

        if exec_res and hasattr(exec_res, 'result') and exec_res.result.error_code.val == MoveItErrorCodes.SUCCESS:
            return "SUCCESS"
        elif exec_res and hasattr(exec_res, 'error_code') and exec_res.error_code.val == MoveItErrorCodes.SUCCESS:
            return "SUCCESS"
        return self._move_above_xy(curr_x, curr_y, target_z)

    def home(self) -> str:
        self.logger.info("Thực hiện: home() [Tư thế cùi chỏ giơ cao]")
        if self.attached_object:
            self.logger.info(f"-> Tháo gỡ liên kết vật lý cho {self.attached_object} trước khi về home...")
            req = DetachLink.Request()
            req.model1_name = 'ur3_robot'
            req.link1_name = 'wrist_3_link'
            req.model2_name = self.attached_object
            req.link2_name = 'link'
            self._detach_client.call_async(req)
            self.attached_object = None

        # Nâng tay lên tầng an toàn trước khi quay về home để tránh quét ngang bàn
        try:
            t = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
            if t.transform.translation.z < (self.approach_z - 0.03):
                self.move_vertical(self.approach_z)
        except Exception:
            pass
        safe_home = self.m.get('home_joints', [0.0, -1.9, 1.0, -1.37, -1.57, 0.0])
        res = self._send_joint_goal(safe_home)
        self._init_planning_scene()
        return res

    # ------------------------------------------------------------ GRIPPER
    def open_gripper(self) -> str:
        self.logger.info("Mở gripper...")
        if self.attached_object:
            self.logger.info(f"-> Tháo gỡ liên kết vật lý cho {self.attached_object}...")
            req = DetachLink.Request()
            req.model1_name = 'ur3_robot'
            req.link1_name = 'wrist_3_link'
            req.model2_name = self.attached_object
            req.link2_name = 'link'
            self._detach_client.call_async(req)
            self.attached_object = None
            self.spin_sleep(0.3)

        msg = Float64MultiArray()
        msg.data = [self.open_val, self.open_val]
        self._gripper_pub.publish(msg)
        self.spin_sleep(0.8)
        return "SUCCESS"

    def close_gripper(self, target_object: str = None) -> str:
        self.logger.info("Khép gripper kẹp khối...")
        msg = Float64MultiArray()
        # Khối hộp dày 4cm (mỗi bên 0.020m). Lệnh 0.0185m mỗi bên ép chặt
        msg.data = [0.0185, 0.0185]
        self._gripper_pub.publish(msg)
        self.spin_sleep(1.2)

        # Publish lần 2 để đảm bảo controller nhận lệnh
        self._gripper_pub.publish(msg)
        self.spin_sleep(0.8)

        gap = self._current_gap
        self.logger.info(f"Độ mở hai ngón kẹp thực tế: {gap*100:.1f} cm")
        # Khối hộp 4cm (0.04m). Khi kẹp vật lý trúng khối, hai ngón kẹp bị chặn ở khoảng 3.5cm - 4.3cm
        if not (0.035 <= gap <= 0.043):
            if gap < 0.035:
                self.logger.warn("Cảnh báo: Hai ngón kẹp khép quá sâu, không phát hiện khối hộp ở giữa!")
            else:
                self.logger.warn("Cảnh báo: Ngón kẹp chưa khép lại đủ!")
            return "GRASP_FAILED"

        if target_object:
            self.logger.info(f"-> Kích hoạt liên kết vật lý (FixedJoint) giữa gripper và '{target_object}'...")
            req = AttachLink.Request()
            req.model1_name = 'ur3_robot'
            req.link1_name = 'wrist_3_link'
            req.model2_name = target_object
            req.link2_name = 'link'
            fut = self._attach_client.call_async(req)
            rclpy.spin_until_future_complete(self.node, fut, timeout_sec=2.0)
            self.attached_object = target_object
            self.spin_sleep(0.3)

        return "SUCCESS"

    # ------------------------------------------------------------ TASK SKILLS
    def pick(self, object_name: str) -> str:
        self.logger.info(f"Thực hiện: pick({object_name})")
        if object_name not in self.objects:
            return "INVALID_OBJECT"

        pos = self.objects[object_name]

        # 1. Mở ngón kẹp rộng 9cm trước khi hạ
        self.open_gripper()
        self._scene_remove(object_name)

        # 2. Tiếp cận đỉnh khối ở tầng an toàn
        if self._move_above_xy(pos['x'], pos['y'], self.approach_z) != "SUCCESS":
            return "PLANNING_FAILED"

        # 3. Hạ thẳng đứng xuống đúng z_grasp để ngón kẹp ôm trọn thân khối
        self.spin_sleep(0.3)
        if self.move_vertical(self.grasp_z) != "SUCCESS":
            return "FAILED"

        # 4. Chờ ổn định trước khi kẹp
        self.spin_sleep(0.3)

        # 5. Kẹp chặt khối bằng ngón tay đệm cao su
        if self.close_gripper(target_object=object_name) != "SUCCESS":
            self.open_gripper()
            self.move_vertical(self.approach_z)
            self._init_planning_scene()
            return "GRASP_FAILED"

        # 6. Đính kèm vật thể vào gripper trong MoveIt planning scene
        self._scene_attach(object_name)
        self.attached_object = object_name
        self.spin_sleep(0.3)

        # 7. Rút thẳng đứng lên tầng cao mang theo khối hộp
        if self.move_vertical(self.approach_z) != "SUCCESS":
            return "FAILED"

        return "SUCCESS"

    def place(self, object_name: str, destination: str) -> str:
        self.logger.info(f"Thực hiện: place({object_name}, {destination})")
        target_pos = self.zones.get(destination) or self.staging_positions.get(destination)
        if object_name not in self.objects or not target_pos:
            return "INVALID_OBJECT"

        # 1. Bay ngang ở tầng an toàn đến trên đỉnh zone
        if self._move_above_xy(target_pos['x'], target_pos['y'], self.approach_z) != "SUCCESS":
            return "PLANNING_FAILED"

        # 2. Hạ thẳng đứng xuống đặt khối lên mặt bàn
        self.spin_sleep(0.2)
        if self.move_vertical(self.place_z) != "SUCCESS":
            return "FAILED"

        # 3. Mở ngón kẹp thả khối (ngắt bám dính cao su, mở gripper, khối tự đứng trên mặt bàn)
        self.open_gripper()
        self._scene_remove(object_name)
        self.attached_object = None
        self.objects[object_name] = {'x': target_pos['x'], 'y': target_pos['y'], 'z': 0.06}
        self.spin_sleep(0.5)

        # 4. Rút thẳng đứng lên tầng an toàn
        res = self.move_vertical(self.approach_z)
        # Thêm lại collision object sau khi gripper đã rút lên cao
        self._scene_detach(object_name, target_pos['x'], target_pos['y'])
        if res != "SUCCESS":
            return "FAILED"

        return "SUCCESS"