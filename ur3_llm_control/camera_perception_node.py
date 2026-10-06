import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import numpy as np
import json
import os
import yaml
from ament_index_python.packages import get_package_share_directory

class CameraPerceptionNode(Node):
    
    def __init__(self):
        super().__init__('camera_perception_node')
        self.bridge = CvBridge()

        # Đọc cấu hình các zone và staging positions tự động từ share directory
        self.zones, self.staging_positions = self._load_scene_config()

        # QoS cho cảm biến Gazebo
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Đăng ký nhận hình ảnh từ Camera Gazebo
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            sensor_qos
        )

        # Publisher công bố trạng thái môi trường
        self.state_pub = self.create_publisher(String, '/world_state_detected', 10)

        # Dải màu HSV cho 5 khối hộp
        self.color_ranges = {
            'red_cube': [
                (np.array([0, 120, 50]), np.array([10, 255, 255])),
                (np.array([170, 120, 50]), np.array([180, 255, 255]))
            ],
            'yellow_cube': [
                (np.array([20, 120, 80]), np.array([35, 255, 255]))
            ],
            'green_cube': [
                (np.array([35, 80, 40]), np.array([85, 255, 255]))
            ],
            'blue_cube': [
                (np.array([100, 120, 50]), np.array([130, 255, 255]))
            ],
            'purple_cube': [
                (np.array([135, 80, 40]), np.array([165, 255, 255]))
            ]
        }

        self.cam_x = 0.38
        self.cam_y = 0.0
        self.cam_z = 1.20
        self.img_w = 640
        self.img_h = 480
        self.fov = 1.15

        self.get_logger().info("Camera Perception Node đã sẵn sàng nhận ảnh từ /camera/image_raw")

    def _load_scene_config(self):
        try:
            pkg_share = get_package_share_directory('ur3_llm_control')
            config_path = os.path.join(pkg_share, 'config', 'scene.yaml')
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
                return data.get('zones', {}), data.get('staging_positions', {})
        except Exception as e:
            self.get_logger().warn(f"Không nạp được scene.yaml, dùng cấu hình mặc định: {e}")
            zones = {
                'zone_a': {'x': 0.42, 'y': -0.16},
                'zone_b': {'x': 0.42, 'y': 0.0},
                'zone_c': {'x': 0.42, 'y': 0.16}
            }
            staging = {
                'temp_pos_1': {'x': 0.45, 'y': -0.08},
                'temp_pos_2': {'x': 0.45, 'y': 0.08}
            }
            return zones, staging

    def image_callback(self, msg):
        self.get_logger().info("Đang xử lý frame ảnh từ Gazebo...", throttle_duration_sec=2.0)
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Lỗi cv_bridge: {e}")
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected_objects = {}

        # Tâm khối cube nằm ở độ cao 0.06m
        cube_center_z = 0.06
        height_diff = self.cam_z - cube_center_z
        span_x = 2.0 * height_diff * np.tan(self.fov / 2.0)
        meters_per_pixel = span_x / self.img_w

        for color_name, ranges in self.color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for (lower, upper) in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(c)
                # Kích thước khối 4x4cm nhìn từ trên xuống tương ứng diện tích ~150 - 800 px
                if 100 < area < 1500:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        dx = -(cy - (self.img_h / 2.0)) * meters_per_pixel
                        dy = -(cx - (self.img_w / 2.0)) * meters_per_pixel

                        real_x = round(self.cam_x + dx, 3)
                        real_y = round(self.cam_y + dy, 3)

                        detected_objects[color_name] = {'x': real_x, 'y': real_y, 'z': 0.06}

        # Đối chiếu vật thể với các vùng zone và staging positions
        zones_status = {}
        for z_name, z_coord in self.zones.items():
            occupant = None
            for obj_name, obj_coord in detected_objects.items():
                dist = np.hypot(obj_coord['x'] - z_coord['x'], obj_coord['y'] - z_coord['y'])
                if dist < 0.08:
                    occupant = obj_name
                    break
            zones_status[z_name] = occupant if occupant else "empty"

        staging_status = {}
        for s_name, s_coord in self.staging_positions.items():
            occupant = None
            for obj_name, obj_coord in detected_objects.items():
                dist = np.hypot(obj_coord['x'] - s_coord['x'], obj_coord['y'] - s_coord['y'])
                if dist < 0.08:
                    occupant = obj_name
                    break
            staging_status[s_name] = occupant if occupant else "empty"

        world_status = {
            "objects": detected_objects,
            "zones_status": zones_status,
            "staging_status": staging_status
        }

        status_msg = String()
        status_msg.data = json.dumps(world_status)
        self.state_pub.publish(status_msg)

        # Log trực tiếp số lượng vật nhận diện được
        num_found = len(detected_objects)
        self.get_logger().info(f"Đã phát hiện {num_found}/5 blocks. Đã publish lên /world_state_detected", throttle_duration_sec=2.0)

def main(args=None):
    rclpy.init(args=args)
    node = CameraPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()