class TaskValidator:
    # Các skill và đối tượng được phép theo đề bài
    ALLOWED_SKILLS = {"home", "pick", "place", "move_above", "open_gripper", "close_gripper"}
    ALLOWED_OBJECTS = {"red_cube", "yellow_cube", "blue_cube"}
    ALLOWED_ZONES = {"zone_a", "zone_b", "zone_c"}

    @classmethod
    def validate_plan(cls, plan_dict: dict) -> tuple[bool, str]:
        """
        Kiểm tra tính hợp lệ của toàn bộ kế hoạch JSON từ LLM.
        Trả về: (is_valid: bool, error_message: str)
        """
        if not isinstance(plan_dict, dict) or "plan" not in plan_dict:
            return False, "Thiếu khóa gốc 'plan' trong JSON."

        steps = plan_dict.get("plan")
        if not isinstance(steps, list) or len(steps) == 0:
            return False, "Kế hoạch 'plan' rỗng hoặc không phải định dạng danh sách."

        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                return False, f"Bước {idx + 1} không phải là một đối tượng dictionary hợp lệ."

            skill = step.get("skill")
            if not skill or skill not in cls.ALLOWED_SKILLS:
                return False, f"Bước {idx + 1}: Skill '{skill}' không nằm trong danh sách cho phép."

            # Kiểm tra tham số theo từng loại skill
            if skill == "pick":
                obj = step.get("object")
                if not obj or obj not in cls.ALLOWED_OBJECTS:
                    return False, f"Bước {idx + 1}: Vật thể '{obj}' không hợp lệ cho lệnh pick()."

            elif skill == "place":
                obj = step.get("object")
                zone = step.get("zone")
                if not obj or obj not in cls.ALLOWED_OBJECTS:
                    return False, f"Bước {idx + 1}: Vật thể '{obj}' không hợp lệ cho lệnh place()."
                if not zone or zone not in cls.ALLOWED_ZONES:
                    return False, f"Bước {idx + 1}: Vùng '{zone}' không hợp lệ cho lệnh place()."

            elif skill == "move_above":
                target = step.get("object") or step.get("zone")
                if not target or (target not in cls.ALLOWED_OBJECTS and target not in cls.ALLOWED_ZONES):
                    return False, f"Bước {idx + 1}: Mục tiêu '{target}' không hợp lệ cho move_above()."

        return True, "Kế hoạch hợp lệ."