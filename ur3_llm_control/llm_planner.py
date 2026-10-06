import json
import os
import re
import yaml
from openai import OpenAI

class LLMPlanner:
    def __init__(self, api_key: str = None, base_url: str = "http://localhost:20128/v1", model: str = "ag/gemini-3.8-flash"):
        self.api_key = api_key or os.getenv("NINEROUTER_API_KEY", "sk-a9b194e71073980b-mb033z-8a5b2133")
        self.base_url = base_url
        self.model = model
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        cfg_path = '/home/gdon/ur_ws/src/ur3_llm_control/config/student_config.yaml'
        self.student_rule = self._load_student_config(cfg_path)

    def _load_student_config(self, path):
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
                return data.get('rule', {})
        except Exception:
            # Quy tắc mặc định theo mã sinh viên
            return {"zone_a": "blue_cube", "zone_b": "yellow_cube", "zone_c": "red_cube"}

    def _build_system_prompt(self, current_state: str = "") -> str:
        """Tạo system prompt động kèm theo trạng thái vật thể thời gian thực"""
        state_section = f"\nCURRENT WORLD STATE:\n{current_state}\n" if current_state else "\nCURRENT WORLD STATE: All target zones (zone_a, zone_b, zone_c) are currently EMPTY.\n"
        
        return f"""You are an intelligent task planner for a UR3 robotic arm equipped with a parallel gripper.
Convert natural language user commands (Vietnamese or English) into a structured, executable JSON action sequence.

Available skills:
- home()
- pick(object)
- place(object, destination)

Allowed objects: red_cube, yellow_cube, blue_cube, green_cube, purple_cube
Allowed destinations: 
- Target Zones: zone_a, zone_b, zone_c
- Temporary Staging Positions (use when clearing obstacles): temp_pos_1, temp_pos_2

LANGUAGE & OBJECT MAPPINGS:
- "do", "đỏ", "red" -> red_cube
- "vang", "vàng", "yellow" -> yellow_cube
- "xanh bien", "xanh dương", "blue" -> blue_cube
- "xanh la", "xanh lục", "green" -> green_cube
- "tim", "tím", "purple" -> purple_cube
- "a", "khu a", "vung a", "zone a" -> zone_a
- "b", "khu b", "vung b", "zone b" -> zone_b
- "c", "khu c", "vung c", "zone c" -> zone_c

STUDENT TARGET RULES:
- zone_a: {self.student_rule.get('zone_a', 'blue_cube')}
- zone_b: {self.student_rule.get('zone_b', 'yellow_cube')}
- zone_c: {self.student_rule.get('zone_c', 'red_cube')}
{state_section}
INTELLIGENCE & PLANNING RULES:
1. Output ONLY a valid JSON object matching the schema. Do NOT include markdown code blocks, backticks, or natural explanations.
2. User Command Priority: If the user specifies an action for a specific block (e.g. "tim vao a", "cho khối tím vào A", "Put yellow cube in zone_a"):
   - ONLY operate on that requested block! Do NOT move other blocks unless the target zone is ALREADY occupied according to CURRENT WORLD STATE.
   - For example: for command "tim vao a", if zone_a is empty, the plan MUST be: pick(purple_cube) -> place(purple_cube, zone_a) -> home().
3. Obstacle Handling:
   - ONLY if CURRENT WORLD STATE explicitly indicates that the destination zone is occupied by another block (occupant != 'empty'):
     * Pick that occupying block and place it into temp_pos_1 (or temp_pos_2).
     * Then pick the target block and place it into the destination zone.
   - If CURRENT WORLD STATE shows the destination zone is empty, do NOT pick or move any obstacle block!
4. Idempotency: If an object is already at its intended destination according to CURRENT WORLD STATE, do NOT move it.
5. If the user asks to "sort", "arrange all", or "follow student ID", plan the sequence to satisfy all student target rules.
6. Always append {{"skill": "home"}} as the final step of the entire plan.

JSON Schema format:
{{
  "plan": [
    {{"skill": "pick", "object": "block_name"}},
    {{"skill": "place", "object": "block_name", "destination": "target_zone_or_temp_pos"}},
    {{"skill": "home"}}
  ]
}}
"""

    def plan(self, user_command: str, current_state: str = "") -> dict:
        try:
            system_prompt = self._build_system_prompt(current_state)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_command}
                ],
                temperature=0.0
            )
            raw_text = response.choices[0].message.content.strip()
            
            # Làm sạch chuỗi JSON an toàn nếu LLM trả về kèm markdown
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                cleaned_text = match.group(0)
            else:
                cleaned_text = raw_text

            return json.loads(cleaned_text)
        except Exception as e:
            # Nếu mất kết nối LLM server, tự động kích hoạt bộ lập kế hoạch luật dự phòng
            fallback = self._rule_based_fallback(user_command, current_state)
            if "plan" in fallback:
                return fallback
            return {"error": str(e), "raw": raw_text if 'raw_text' in locals() else ""}

    def _rule_based_fallback(self, user_command: str, current_state: str = "") -> dict:
        """Kế hoạch dự phòng thông minh khi mất kết nối mạng hoặc LLM server ngoại tuyến"""
        cmd = user_command.lower().strip()
        obj_map = [
            ("purple_cube", ["tim", "tím", "purple"]),
            ("red_cube", ["do", "đỏ", "red"]),
            ("yellow_cube", ["vang", "vàng", "yellow"]),
            ("blue_cube", ["xanh bien", "xanh dương", "blue"]),
            ("green_cube", ["xanh la", "xanh lục", "green"]),
        ]
        dest_map = [
            ("zone_a", ["zone a", "zone_a", "khu a", "vung a", "vao a", "đến a", " a"]),
            ("zone_b", ["zone b", "zone_b", "khu b", "vung b", "vao b", "đến b", " b"]),
            ("zone_c", ["zone c", "zone_c", "khu c", "vung c", "vao c", "đến c", " c"]),
        ]

        target_obj = None
        for obj, kw_list in obj_map:
            if any(kw in cmd for kw in kw_list):
                target_obj = obj
                break

        target_dest = None
        for dest, kw_list in dest_map:
            if any(kw in cmd for kw in kw_list):
                target_dest = dest
                break

        if not target_obj or not target_dest:
            return {"error": f"Không thể nhận diện mục tiêu từ lệnh: '{user_command}'"}

        plan = []
        if current_state:
            try:
                st = json.loads(current_state) if isinstance(current_state, str) else current_state
                occupant = st.get("zones_status", {}).get(target_dest, "empty")
                if occupant != "empty" and occupant != target_obj:
                    plan.append({"skill": "pick", "object": occupant})
                    plan.append({"skill": "place", "object": occupant, "destination": "temp_pos_1"})
            except Exception:
                pass

        plan.append({"skill": "pick", "object": target_obj})
        plan.append({"skill": "place", "object": target_obj, "destination": target_dest})
        plan.append({"skill": "home"})
        return {"plan": plan}


if __name__ == '__main__':
    # Test thử trực tiếp khả năng giải phóng vật cản
    planner = LLMPlanner()
    
    # Giả lập trạng thái camera nhận diện: zone_b đang bị red_cube chiếm chỗ
    dummy_state = json.dumps({
        "objects": {
            "yellow_cube": {"x": 0.36, "y": 0.0},
            "red_cube": {"x": 0.05, "y": 0.38} # Nằm đúng zone_b
        },
        "zones_status": {
            "zone_a": "empty",
            "zone_b": "red_cube", # Bị chiếm
            "zone_c": "empty"
        }
    })
    
    print("--- TEST PROMPT GIẢI PHÓNG VẬT CẢN ---")
    test_cmd = "Hãy đặt khối vàng vào zone_b"
    result = planner.plan(test_cmd, current_state=dummy_state)
    print(json.dumps(result, indent=2, ensure_ascii=False))