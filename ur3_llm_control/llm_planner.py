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
            return {"zone_a": "blue_cube", "zone_b": "yellow_cube", "zone_c": "red_cube"}

    def _build_system_prompt(self, current_state: str = "") -> str:
        """Tạo system prompt động kèm theo trạng thái vật thể thời gian thực"""
        state_section = f"\nCURRENT WORLD STATE:\n{current_state}\n" if current_state else ""
        
        return f"""You are an intelligent task planner for a UR3 robotic arm.
Convert natural language commands into a structured JSON execution plan.

Available skills:
- home()
- pick(object)
- place(object, zone)

Allowed objects: red_cube, yellow_cube, blue_cube
Allowed zones: zone_a, zone_b, zone_c

STUDENT ID TARGET RULES:
- zone_a target: {self.student_rule.get('zone_a', 'blue_cube')}
- zone_b target: {self.student_rule.get('zone_b', 'yellow_cube')}
- zone_c target: {self.student_rule.get('zone_c', 'red_cube')}
{state_section}
INTELLIGENCE RULES:
1. Output ONLY a valid JSON object. No explanations or markdown tags.
2. Each zone can only hold ONE object at a time.
3. If a target zone is already occupied by the WRONG object, plan to move that wrong object out first before placing the new object.
4. If an object is already located at its correct target zone according to CURRENT WORLD STATE, do NOT move it again.
5. If the user gives a single-step command (e.g. "Put the red cube in zone B"), plan only for that command.
6. If the user asks to "arrange all objects" or "arrange according to student ID", plan necessary pick-and-place sequences to achieve the student targets.
7. Format schema:
{{
  "plan": [
    {{"skill": "pick", "object": "..."}},
    {{"skill": "place", "object": "...", "zone": "..."}},
    {{"skill": "home"}}
  ]
}}
8. Always end the entire plan with {{"skill": "home"}}.
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
            
            # Làm sạch chuỗi JSON an toàn
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                cleaned_text = match.group(0)
            else:
                cleaned_text = raw_text

            return json.loads(cleaned_text)
        except Exception as e:
            return {"error": str(e)}