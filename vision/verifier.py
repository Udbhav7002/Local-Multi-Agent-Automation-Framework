"""
Vision verification module for evaluating screen states using LLaVA.
"""
import asyncio
import json
import re

import ollama

from core.logger import setup_logger
from vision.screen_capture import ScreenCapture

logger = setup_logger("VisionVerifier")


class VisionVerifier: # pylint: disable=too-few-public-methods
    """
    Uses a vision-language model to determine if a goal was achieved
    on the current screen.
    """
    def __init__(self, model_name: str = "llava") -> None:
        self.model_name = model_name
        self.screen_cap = ScreenCapture()

    async def verify(self, expected_outcome: str) -> tuple[bool, str]:
        """
        Takes a screenshot and asks the model if the outcome was achieved.
        """
        logger.info("Capturing screen to verify: '%s'...", expected_outcome)
        screenshot_path = self.screen_cap.capture_fullscreen(
            "current_state.png")

        prompt = (
            f"Look at this screenshot. Was this outcome achieved: '{expected_outcome}'?\n"
            "Note: If the outcome involves opening a specific web browser (like Brave), "
            "and you see ANY web browser open, consider it a success. "
            "You must reply with a valid JSON object exactly like this: "
            "{\"success\": true, \"reason\": \"your reason here\"}"
        )

        try:
            logger.debug("Asking %s to verify...", self.model_name)
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [screenshot_path]
                }],
                format='json'
            )

            answer = response.get('message', {}).get('content', '').strip()

            # Clean up markdown code blocks if present
            clean_answer = answer.replace(
                "```json", "").replace(
                "```", "").strip()

            try:
                # Try to parse the cleaned string directly
                result = json.loads(clean_answer)
                is_success = result.get('success', False)
                reason = result.get('reason', 'No reason provided')
                logger.info(
                    "Verifier response: %s (Success: %s)", reason, is_success)
                return is_success, reason
            except json.JSONDecodeError:
                # Fallback: Extract via regex if JSON is malformed
                success_match = re.search(
                    r'"success"\s*:\s*(true|false)', answer, re.IGNORECASE)
                reason_match = re.search(
                    r'"reason"\s*:\s*"([^"]+)"', answer, re.IGNORECASE)

                if success_match:
                    is_success = success_match.group(1).lower() == 'true'
                    reason = reason_match.group(
                        1) if reason_match else 'Failed to parse reason.'
                    logger.warning(
                        "Verifier response (Regex Fallback): %s (Success: %s)",
                        reason, is_success)
                    return is_success, reason

            # Ultimate Fallback
            logger.error("Verifier returned invalid format: %s", answer)
            return False, f"Invalid vision format: {answer}"

        except Exception as e: # pylint: disable=broad-exception-caught
            logger.error("Vision verification failed due to error: %s", e)
            return False, f"Vision verification failed due to error: {e}"
