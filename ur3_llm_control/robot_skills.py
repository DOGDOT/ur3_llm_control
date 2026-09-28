import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from visualization_msgs.msg import Marker, MarkerArray
import yaml
import os
import math
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints, 
    PositionConstraint, 
    OrientationConstraint, 
    JointConstraint,
    PlanningScene,
    CollisionObject,
    MoveItErrorCodes
)
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener

class RobotSkills:
    def __init__(self, node: Node, config_path: str = None):
        self.node = node
        self.logger = node.get_logger()
        
        # Đọc cấu hình scene.yaml
        if config_path is None:
            config_path = '/home/gdon/ur_ws/src/ur3_llm_control/config/scene.yaml'
        
        self.config = self._load_config(config_path)
        self.objects = self.config.get('objects', {})
        self.zones = self.config.get('zones', {})

        # Khởi tạo Action Client MoveIt 2
        self._move_group_client = ActionClient(self.node, MoveGroup, 'move_action')
        self.logger.info("Đang chờ MoveGroup Action Server...")
        if not self._move_group_client.wait_for_server(timeout_sec=5.0):
            self.logger.warn("Chưa tìm thấy MoveGroup Action Server. Hãy chắc chắn đã bật MoveIt!")

        # Publisher để thêm mặt sàn/bàn chống va chạm vào Planning Scene
        self._scene_pub = self.node.create_publisher(PlanningScene, '/planning_scene', 10)

        self.attached_object = None

        # Khởi tạo bộ lắng nghe TF để theo dõi vị trí thực tế của đầu gắp
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

        self._marker_pub = self.node.create_publisher(MarkerArray, '/visualization_marker_array', 10)
        self._init_scene_objects()
        
        # Tạo Timer 20Hz (mỗi 0.05s) tự động phát lại Marker bám dính
        self._marker_timer = self.node.create_timer(0.05, self._publish_markers_timer)

    def _load_config(self, path):
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
                return data if data is not None else {}
        except Exception as e:
            self.logger.error(f"Không thể đọc config: {e}")
            return {}

    def _init_scene_objects(self):
        """Tạo sàn và cập nhật Marker hiển thị (hỗ trợ bám dính theo đầu gắp)"""
        scene_msg = PlanningScene()
        scene_msg.is_diff = True

        floor_obj = CollisionObject()
        floor_obj.header.frame_id = "base_link"
        floor_obj.id = "floor_table"
        floor_box = SolidPrimitive()
        floor_box.type = SolidPrimitive.BOX
        floor_box.dimensions = [1.6, 1.6, 0.02]
        
        floor_pose = PoseStamped()
        floor_pose.header.frame_id = "base_link"
        floor_pose.pose.position.z = -0.01
        floor_pose.pose.orientation.w = 1.0
        floor_obj.primitives.append(floor_box)
        floor_obj.primitive_poses.append(floor_pose.pose)
        floor_obj.operation = CollisionObject.ADD
        scene_msg.world.collision_objects.append(floor_obj)
        self._scene_pub.publish(scene_msg)

        marker_array = MarkerArray()

        cube_colors = {
            'red_cube': (0.9, 0.1, 0.1),
            'yellow_cube': (0.9, 0.8, 0.1),
            'blue_cube': (0.1, 0.3, 0.9),
        }
        cube_id_map = {'red_cube': 1, 'yellow_cube': 2, 'blue_cube': 3}
        zone_id_map = {'zone_a': 11, 'zone_b': 12, 'zone_c': 13}

        for obj_name, coords in self.objects.items():
            m = Marker()
            m.header.stamp = self.node.get_clock().now().to_msg()
            m.ns = "cubes"
            m.id = cube_id_map.get(obj_name, 0)
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.scale.x = 0.04
            m.scale.y = 0.04
            m.scale.z = 0.04
            
            rgb = cube_colors.get(obj_name, (0.5, 0.5, 0.5))
            m.color.r = rgb[0]
            m.color.g = rgb[1]
            m.color.b = rgb[2]
            m.color.a = 1.0

            m.header.frame_id = "base_link"
            m.pose.position.x = float(coords['x'])
            m.pose.position.y = float(coords['y'])
            m.pose.position.z = float(coords['z'])
            m.pose.orientation.w = 1.0
            marker_array.markers.append(m)

        for zone_name, coords in self.zones.items():
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp = self.node.get_clock().now().to_msg()
            m.ns = "zones"
            m.id = zone_id_map.get(zone_name, 10)
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(coords['x'])
            m.pose.position.y = float(coords['y'])
            m.pose.position.z = 0.002
            m.pose.orientation.w = 1.0
            m.scale.x = 0.10
            m.scale.y = 0.10
            m.scale.z = 0.004
            m.color.r = 0.2
            m.color.g = 0.8
            m.color.b = 0.3
            m.color.a = 0.4
            marker_array.markers.append(m)

        for _ in range(3):
            self._marker_pub.publish(marker_array)

    def _send_pose_goal(self, x, y, z, planning_group="ur_manipulator", constrain_orientation=False):
        """Gửi mục tiêu vị trí Cartesian có hỗ trợ bật/tắt ràng buộc hướng"""
        if not self._move_group_client.server_is_ready():
            return "FAILED"

        self._init_scene_objects()

        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = planning_group
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 0.4
        goal_msg.request.max_acceleration_scaling_factor = 0.4

        c = Constraints()
        
        # 1. Ràng buộc vị trí
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = "base_link"
        pos_constraint.link_name = "wrist_3_link"

        shape = SolidPrimitive()
        shape.type = SolidPrimitive.SPHERE
        shape.dimensions = [0.02]
        pos_constraint.constraint_region.primitives.append(shape)
        
        target_pose = PoseStamped()
        target_pose.header.frame_id = "base_link"
        target_pose.pose.position.x = float(x)
        target_pose.pose.position.y = float(y)
        target_pose.pose.position.z = float(z)
        pos_constraint.constraint_region.primitive_poses.append(target_pose.pose)
        pos_constraint.weight = 1.0
        c.position_constraints.append(pos_constraint)

        # 2. Ràng buộc hướng (nếu bật)
        if constrain_orientation:
            orient_constraint = OrientationConstraint()
            orient_constraint.header.frame_id = "base_link"
            orient_constraint.link_name = "wrist_3_link"
            orient_constraint.orientation.x = 1.0
            orient_constraint.orientation.y = 0.0
            orient_constraint.orientation.z = 0.0
            orient_constraint.orientation.w = 0.0
            orient_constraint.absolute_x_axis_tolerance = 1.2
            orient_constraint.absolute_y_axis_tolerance = 1.2
            orient_constraint.absolute_z_axis_tolerance = 3.14
            orient_constraint.weight = 0.36
            c.orientation_constraints.append(orient_constraint)

        goal_msg.request.goal_constraints.append(c)

        send_future = self._move_group_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self.node, send_future, timeout_sec=6.0)
        goal_handle = send_future.result()

        if not goal_handle or not goal_handle.accepted:
            return "PLANNING_FAILED"

        res_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, res_future, timeout_sec=15.0)
        result = res_future.result()

        if result and result.result.error_code.val == MoveItErrorCodes.SUCCESS:
            return "SUCCESS"
        else:
            return "FAILED"

    def _send_joint_goal(self, joint_positions, planning_group="ur_manipulator"):
        """Gửi mục tiêu không gian khớp (Joint Space) cho lệnh Home hoặc tư thế trung gian"""
        if not self._move_group_client.server_is_ready():
            return "FAILED"

        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = planning_group
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 0.4
        goal_msg.request.max_acceleration_scaling_factor = 0.4

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

        send_future = self._move_group_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self.node, send_future, timeout_sec=6.0)
        goal_handle = send_future.result()

        if not goal_handle or not goal_handle.accepted:
            return "PLANNING_FAILED"

        res_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, res_future, timeout_sec=15.0)
        result = res_future.result()

        if result and result.result.error_code.val == MoveItErrorCodes.SUCCESS:
            return "SUCCESS"
        else:
            return "FAILED"

    # ================= CÁC SKILL ROBOT ================= #

    def home(self) -> str:
        self.logger.info("Thực hiện skill: home()")
        try:
            t = self.tf_buffer.lookup_transform(
                'base_link',
                'wrist_3_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            curr_x = t.transform.translation.x
            curr_y = t.transform.translation.y
            curr_z = t.transform.translation.z

            if curr_z < 0.20:
                self.logger.info("Đang nhấc tay lên cao an toàn trước khi về Home...")
                self._send_pose_goal(curr_x, curr_y, 0.25, constrain_orientation=False)
        except Exception as e:
            self.logger.warn(f"Không thể đọc TF trước khi về Home: {e}")

        safe_home_joints = [0.0, -1.57, 1.2, -1.2, -1.57, 0.0]
        return self._send_joint_goal(safe_home_joints)

    def move_above(self, target_name: str) -> str:
        coords = self.objects.get(target_name) or self.zones.get(target_name)
        if not coords:
            return "INVALID_OBJECT"
        return self._send_pose_goal(coords['x'], coords['y'], max(0.20, coords['z'] + 0.15), constrain_orientation=False)

    def open_gripper(self) -> str:
        self.logger.info("Thực hiện skill: open_gripper()")
        return "SUCCESS"

    def close_gripper(self) -> str:
        self.logger.info("Thực hiện skill: close_gripper()")
        return "SUCCESS"

    def _publish_markers_timer(self):
        if self.attached_object and self.attached_object in self.objects:
            try:
                t = self.tf_buffer.lookup_transform(
                    'base_link',
                    'wrist_3_link',
                    rclpy.time.Time()
                )
                self.objects[self.attached_object]['x'] = t.transform.translation.x
                self.objects[self.attached_object]['y'] = t.transform.translation.y
                self.objects[self.attached_object]['z'] = max(0.02, t.transform.translation.z - 0.05)
            except Exception:
                pass
        self._publish_markers()

    def _publish_markers(self):
        marker_array = MarkerArray()
        cube_colors = {
            'red_cube': (0.9, 0.1, 0.1),
            'yellow_cube': (0.9, 0.8, 0.1),
            'blue_cube': (0.1, 0.3, 0.9),
        }
        cube_id_map = {'red_cube': 1, 'yellow_cube': 2, 'blue_cube': 3}
        zone_id_map = {'zone_a': 11, 'zone_b': 12, 'zone_c': 13}

        for obj_name, coords in self.objects.items():
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp = self.node.get_clock().now().to_msg()
            m.ns = "cubes"
            m.id = cube_id_map.get(obj_name, 0)
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(coords['x'])
            m.pose.position.y = float(coords['y'])
            m.pose.position.z = float(coords['z'])
            m.pose.orientation.w = 1.0
            m.scale.x = 0.04
            m.scale.y = 0.04
            m.scale.z = 0.04
            
            rgb = cube_colors.get(obj_name, (0.5, 0.5, 0.5))
            m.color.r = rgb[0]
            m.color.g = rgb[1]
            m.color.b = rgb[2]
            m.color.a = 1.0
            marker_array.markers.append(m)

        for zone_name, coords in self.zones.items():
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp = self.node.get_clock().now().to_msg()
            m.ns = "zones"
            m.id = zone_id_map.get(zone_name, 10)
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(coords['x'])
            m.pose.position.y = float(coords['y'])
            m.pose.position.z = 0.002
            m.pose.orientation.w = 1.0
            m.scale.x = 0.10
            m.scale.y = 0.10
            m.scale.z = 0.004
            m.color.r = 0.2
            m.color.g = 0.8
            m.color.b = 0.3
            m.color.a = 0.4
            marker_array.markers.append(m)

        self._marker_pub.publish(marker_array)
    
    def get_world_state_description(self) -> str:
        state_lines = []
        for obj_name, pos in self.objects.items():
            located_zone = None
            for zone_name, z_pos in self.zones.items():
                dist = ((pos['x'] - z_pos['x'])**2 + (pos['y'] - z_pos['y'])**2)**0.5
                if dist < 0.07:
                    located_zone = zone_name
                    break
            
            if located_zone:
                state_lines.append(f"- {obj_name} is currently at {located_zone}.")
            else:
                state_lines.append(f"- {obj_name} is at initial position (not in any zone).")
        return "\n".join(state_lines)

    def pick(self, object_name: str) -> str:
        self.logger.info(f"Thực hiện skill: pick({object_name})")
        if object_name not in self.objects:
            return "INVALID_OBJECT"

        pos = self.objects[object_name]
        z_safe_lift = 0.20
        z_grasp = 0.08

        self.open_gripper()

        status = self._send_pose_goal(pos['x'], pos['y'], z_safe_lift, constrain_orientation=False)
        if status != "SUCCESS":
            return "PLANNING_FAILED"

        status = self._send_pose_goal(pos['x'], pos['y'], z_grasp, constrain_orientation=True)
        if status != "SUCCESS":
            return "FAILED"

        self.close_gripper()
        self.attached_object = object_name

        status = self._send_pose_goal(pos['x'], pos['y'], z_safe_lift, constrain_orientation=False)
        if status != "SUCCESS":
            return "FAILED"

        return "SUCCESS"

    def place(self, object_name: str, zone_name: str) -> str:
        self.logger.info(f"Thực hiện skill: place({object_name}, {zone_name})")
        if object_name not in self.objects or zone_name not in self.zones:
            return "INVALID_OBJECT"

        z_pos = self.zones[zone_name]
        z_safe_lift = 0.20
        z_place = 0.08

        status = self._send_pose_goal(z_pos['x'], z_pos['y'], z_safe_lift, constrain_orientation=False)
        if status != "SUCCESS":
            return "PLANNING_FAILED"

        status = self._send_pose_goal(z_pos['x'], z_pos['y'], z_place, constrain_orientation=True)
        if status != "SUCCESS":
            return "FAILED"

        self.open_gripper()
        self.attached_object = None
        self.objects[object_name] = {'x': z_pos['x'], 'y': z_pos['y'], 'z': 0.04}

        status = self._send_pose_goal(z_pos['x'], z_pos['y'], z_safe_lift, constrain_orientation=False)
        if status != "SUCCESS":
            return "FAILED"

        return "SUCCESS"