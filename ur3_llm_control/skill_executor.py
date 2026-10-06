import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import json
import time

from ur3_llm_control.robot_skills import RobotSkills
from ur3_llm_control.task_validator import TaskValidator
from ur3_llm_control.llm_planner import LLMPlanner

class SkillExecutorNode(Node):
    def __init__(self):
        super().__init__('skill_executor_node')
        self.skills = RobotSkills(self)
        self.validator = TaskValidator()
        self.planner = LLMPlanner()

        self.latest_world_state = {}

        # Lắng nghe trạng thái môi trường thời gian thực từ Camera Perception
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.state_sub = self.create_subscription(
            String,
            '/world_state_detected',
            self._world_state_callback,
            qos
        )

        self.get_logger().info("Skill Executor Node đã khởi tạo sẵn sàng!")

    def _world_state_callback(self, msg: String):
        try:
            self.latest_world_state = json.loads(msg.data)
            # Cập nhật tọa độ động sang robot_skills nếu có hỗ trợ
            if hasattr(self.skills, 'update_detected_objects'):
                self.skills.update_detected_objects(self.latest_world_state.get('objects', {}))
        except Exception as e:
            self.get_logger().error(f"Lỗi parse world state JSON: {e}")

    def get_current_state_string(self) -> str:
        """Định dạng trạng thái môi trường từ Camera Perception thời gian thực cho LLM"""
        if self.latest_world_state:
            return json.dumps(self.latest_world_state, indent=2)
        if hasattr(self.skills, 'get_world_state_description'):
            return self.skills.get_world_state_description()
        return ""

    def execute_command(self, command_text: str):
        print("\n" + "="*50)
        print(f"USER COMMAND:\n{command_text}\n")

        # Thu thập trạng thái môi trường cập nhật thời gian thực từ Camera Perception
        start_wait = time.time()
        while time.time() - start_wait < 3.0 and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_world_state:
                break
        current_state = self.get_current_state_string()

        # 2. Gọi LLM sinh kế hoạch (đã hỗ trợ giải phóng vật cản ra temp_pos)
        plan_data = self.planner.plan(command_text, current_state=current_state)
        if "error" in plan_data:
            print(f"[ERROR] Không thể lấy kế hoạch từ LLM: {plan_data['error']}")
            return

        # 3. Kiểm tra tính hợp lệ qua Validator
        is_valid, msg = self.validator.validate_plan(plan_data)
        if not is_valid:
            print(f"[REJECTED] Kế hoạch không hợp lệ: {msg}")
            return

        # 4. In kế hoạch LLM PLAN ra terminal
        print("LLM PLAN:")
        steps = plan_data.get("plan", [])
        for idx, step in enumerate(steps):
            skill = step.get("skill")
            dest = step.get("destination") or step.get("zone")
            if skill == "home":
                print(f"  {idx + 1}. home()")
            elif skill == "pick":
                print(f"  {idx + 1}. pick({step.get('object')})")
            elif skill == "place":
                print(f"  {idx + 1}. place({step.get('object')}, {dest})")
            elif skill == "move_above":
                tgt = step.get("object") or dest
                print(f"  {idx + 1}. move_above({tgt})")
        print()

        # 5. Thực thi từng Skill tuần tự
        print("EXECUTION:")
        for step in steps:
            skill = step.get("skill")
            dest = step.get("destination") or step.get("zone")
            res = "FAILED"
            step_str = ""

            if skill == "home":
                step_str = "home()"
                res = self.skills.home()
            elif skill == "pick":
                step_str = f"pick({step.get('object')})"
                res = self.skills.pick(step.get("object"))
            elif skill == "place":
                step_str = f"place({step.get('object')}, {dest})"
                res = self.skills.place(step.get("object"), dest)
            elif skill == "move_above":
                tgt = step.get("object") or dest
                step_str = f"move_above({tgt})"
                res = self.skills.move_above(tgt)

            dots = "." * max(2, (35 - len(step_str)))
            print(f"{step_str} {dots} {res}")

            if res != "SUCCESS":
                print(f"\n[TASK FAILED] Dừng thực thi tại bước: {step_str}")
                return

        print("\nTASK SUCCESS")
        print("="*50 + "\n")


def main(args=None):
    rclpy.init(args=args)
    executor = SkillExecutorNode()

    try:
        while rclpy.ok():
            # Spin nhẹ để nhận cập nhật mới nhất từ /world_state_detected
            rclpy.spin_once(executor, timeout_sec=0.1)
            cmd = input("Nhập lệnh (hoặc 'exit' để thoát): ")
            if cmd.strip().lower() == 'exit':
                break
            if cmd.strip():
                executor.execute_command(cmd.strip())
    except KeyboardInterrupt:
        pass
    finally:
        executor.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()