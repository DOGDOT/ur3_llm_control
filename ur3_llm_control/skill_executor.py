import rclpy
from rclpy.node import Node
from ur3_llm_control.robot_skills import RobotSkills
from ur3_llm_control.task_validator import TaskValidator
from ur3_llm_control.llm_planner import LLMPlanner

class SkillExecutorNode(Node):
    def __init__(self):
        super().__init__('skill_executor_node')
        self.skills = RobotSkills(self)
        self.validator = TaskValidator()
        self.planner = LLMPlanner()
        self.get_logger().info("Skill Executor Node đã khởi tạo sẵn sàng!")

    def execute_command(self, command_text: str):
        print("\n" + "="*50)
        print(f"USER COMMAND:\n{command_text}\n")

        # 1. Đọc trạng thái vị trí hiện tại của các cube từ robot_skills
        current_state = ""
        if hasattr(self.skills, 'get_world_state_description'):
            current_state = self.skills.get_world_state_description()

        # 2. Gọi LLM sinh kế hoạch kèm theo ngữ cảnh thực tế
        plan_data = self.planner.plan(command_text, current_state=current_state)
        if "error" in plan_data:
            print(f"[ERROR] Không thể kết nối LLM: {plan_data['error']}")
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
            if skill == "home":
                print(f"  {idx + 1}. home()")
            elif skill == "pick":
                print(f"  {idx + 1}. pick({step.get('object')})")
            elif skill == "place":
                print(f"  {idx + 1}. place({step.get('object')}, {step.get('zone')})")
            elif skill == "move_above":
                tgt = step.get("object") or step.get("zone")
                print(f"  {idx + 1}. move_above({tgt})")
        print()

        # 5. Thực thi từng Skill
        print("EXECUTION:")
        for step in steps:
            skill = step.get("skill")
            res = "FAILED"
            step_str = ""

            if skill == "home":
                step_str = "home()"
                res = self.skills.home()
            elif skill == "pick":
                step_str = f"pick({step.get('object')})"
                res = self.skills.pick(step.get("object"))
            elif skill == "place":
                step_str = f"place({step.get('object')}, {step.get('zone')})"
                res = self.skills.place(step.get("object"), step.get("zone"))
            elif skill == "move_above":
                tgt = step.get("object") or step.get("zone")
                step_str = f"move_above({tgt})"
                res = self.skills.move_above(tgt)

            dots = "." * max(2, (30 - len(step_str)))
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
            cmd = input("Nhập lệnh (hoặc 'exit' để thoát): ")
            if cmd.strip().lower() == 'exit':
                break
            if cmd.strip():
                executor.execute_command(cmd.strip())
    except KeyboardInterrupt:
        pass
    finally:
        executor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()