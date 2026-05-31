"""
Planner module for breaking down user tasks into actionable JSON steps.
"""
import asyncio
import json
import re

import ollama

from core.logger import setup_logger

logger = setup_logger("Planner")


class Planner: # pylint: disable=too-few-public-methods
    """
    Agent that generates a sequential plan of actions for the framework to execute.
    """
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    async def generate_plan(self, task_description: str) -> str:
        """
        Takes a task and outputs a strict JSON array of high-level sub-tasks.
        """
        system_prompt = """
        You are the Manager Agent (Planner) for a local OS controller.
        Your job is to break down the user's task into a strict JSON array of high-level strings.
        These sub-tasks will be passed to a specialized Worker Agent to execute.

        RULES:
        1. Keep the steps high-level and focused on the goal.
        2. Do not specify exact CSS selectors or CLI commands. The Worker handles that.
        3. If you receive `[SYSTEM FEEDBACK]` detailing a failure, DO NOT start from scratch. Output the remaining sub-tasks needed from the current state.
        4. NEVER output a standalone step just to "Open a browser". The system will automatically open the browser when you navigate. Start directly with the web navigation step.
        5. DO NOT guess direct URLs. Always instruct the Worker to navigate to a search engine (like google.com) and search for the target.
        
        OUTPUT FORMAT:
        You must output ONLY valid JSON in this exact format. No markdown blocks.
        [
            "High level step 1",
            "High level step 2"
        ]

        EXAMPLES:
        User: "open brave"
        ["Open the Brave browser"]
        
        User: "search for gemini and click the first link"
        [
            "Open browser and navigate to a search engine",
            "Search for 'gemini ai'",
            "Click the first relevant link for Gemini AI"
        ]
        """

        logger.debug("Generating plan using model %s...", self.model_name)
        try:
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task_description}
                ],
            )
            content = response.get('message', {}).get('content', '').strip()

            # Remove <think>...</think> blocks for reasoning models like DeepSeek-R1
            content = re.sub(
                r'<think>.*?</think>',
                '',
                content,
                flags=re.DOTALL).strip()

            # Extract JSON array robustly using non-greedy regex
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                content = match.group(0)

            # Proactive Fix: Remove trailing commas in JSON (common LLM hallucination)
            content = re.sub(r',\s*([\]}])', r'\1', content)

            # Validation to ensure it's valid JSON
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(
                    "Planner output invalid JSON: %s \nContent: %s", e, content)
                raise ValueError(
                    f"Generated plan is not valid JSON: {e}") from e

            return content.strip()

        except Exception as e: # pylint: disable=broad-exception-caught
            logger.error("Planner failed to generate plan: %s", e)
            raise e
