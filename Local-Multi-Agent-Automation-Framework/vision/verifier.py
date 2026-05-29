import ollama
import asyncio
from vision.screen_capture import ScreenCapture

import json

class VisionVerifier:
    def __init__(self, model_name="llava"):
        self.model_name = model_name
        self.screen_cap = ScreenCapture()

    async def verify(self, expected_outcome: str) -> tuple[bool, str]:
        """
        Uses LLaVA vision model to check if the expected outcome is present on screen.
        """
        print(f"[VISION] Capturing screen to verify: '{expected_outcome}'...")
        screenshot_path = self.screen_cap.capture_fullscreen("current_state.png")
        
        prompt = f"Look at this screenshot. Was this outcome achieved: '{expected_outcome}'? You must reply with a valid JSON object exactly like this: {{\"success\": true, \"reason\": \"your reason here\"}}"
        
        try:
            print(f"[VISION] Asking {self.model_name} to verify...")
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [screenshot_path]
                }]
            )
            
            answer = response['message']['content'].strip()
            
            # Clean up markdown code blocks if present
            clean_answer = answer.replace("```json", "").replace("```", "").strip()
            
            try:
                # Try to parse the cleaned string directly
                result = json.loads(clean_answer)
                is_success = result.get('success', False)
                reason = result.get('reason', 'No reason provided')
                print(f"[VISION] Verifier response: {reason} (Success: {is_success})")
                return is_success, reason
            except json.JSONDecodeError:
                # Fallback: Extract via regex if JSON is malformed (e.g. extra braces)
                import re
                success_match = re.search(r'"success"\s*:\s*(true|false)', answer, re.IGNORECASE)
                reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', answer, re.IGNORECASE)
                
                if success_match:
                    is_success = success_match.group(1).lower() == 'true'
                    reason = reason_match.group(1) if reason_match else 'Failed to parse reason.'
                    print(f"[VISION] Verifier response (Regex Fallback): {reason} (Success: {is_success})")
                    return is_success, reason
                    
            # Ultimate Fallback if everything fails
            print(f"[VISION] Verifier returned invalid format: {answer}")
            return False, f"Invalid vision format: {answer}"
                
        except Exception as e:
            return False, f"Vision verification failed due to error: {str(e)}"
