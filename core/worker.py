"""
Worker module for handling specialized sub-tasks assigned by the Manager (Planner).
"""
import asyncio
import json
import re

import ollama

from core.logger import setup_logger

logger = setup_logger("Worker")


class Worker:
    """
    Executes a specific sub-task given by the Manager.
    Outputs low-level CLI, GUI, or Browser commands in JSON.
    """
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    async def generate_action(self, sub_task: str, current_context: str) -> str:
        """
        Takes a specific sub-task and context, and generates the exact JSON step.
        """
        system_prompt = """
        You are a specialized Worker Agent. Your job is to take a specific sub-task from the Manager and translate it into EXACT executable commands.

        RULES:
        1. YOU MUST PRIORITIZE CLI ACTIONS for system management.
        2. To OPEN LOCAL APPLICATIONS, use "method": "system", "action": "open_app".
        3. CRITICAL BROWSER RULE: If the task involves a web browser AT ALL (e.g. "open brave", "launch browser", or navigating to a website), YOU MUST NEVER use "open_app". You must ALWAYS use "method": "browser" with "action": "goto". If no URL is provided, use "https://google.com" as the target. The framework will automatically launch the correct browser for you in the background. Using "open_app" for browsers breaks the automation!
        4. For WEB TASKS, use "method": "browser" with "action": "goto" for navigation. For interacting with the page (clicking, typing), YOU MUST use "method": "gui" with "action": "click" or "type" to use the computer vision mouse/keyboard system. DO NOT use DOM selectors.
        
        OUTPUT FORMAT:
        You must output ONLY valid JSON in this exact format. No markdown blocks, no other text.
        [
            {
                "action": "run_command" | "click" | "type" | "hotkey" | "goto" | "sleep" | "close_browser" | "open_app",
                "target": "string (command, element name/description, URL, or seconds)",
                "method": "cli" | "gui" | "browser" | "system"
            }
        ]
        """

        prompt = f"Context:\n{current_context}\n\nYour Sub-Task:\n{sub_task}"
        
        logger.debug("Worker (%s) generating action for sub-task: %s", self.model_name, sub_task)
        
        try:
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
            )
            content = response.get('message', {}).get('content', '').strip()

            # Remove <think>...</think>
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                content = match.group(0)

            content = re.sub(r',\s*([\]}])', r'\1', content)

            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                logger.error("Worker output invalid JSON: %s", e)
                raise ValueError(f"Worker generated invalid JSON: {e}") from e

            return content.strip()

        except Exception as e:
            logger.error("Worker failed: %s", e)
            raise e
