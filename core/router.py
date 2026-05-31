"""
Router module for evaluating the intent of a user prompt.
Determines whether the user is asking a question or issuing a task command.
"""
import asyncio
import json
from typing import Optional, Tuple

import ollama

from core.logger import setup_logger

logger = setup_logger("Router")


class Router: # pylint: disable=too-few-public-methods
    """
    Agent that routes user requests to either the QA LLM or the Planner.
    """
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    async def classify(self, user_text: str) -> Tuple[str, Optional[str]]:
        """
        Classifies user text as either 'question' or 'task' using structured JSON.
        Returns a tuple: (intent, answer_if_question)
        """
        # Deterministic Heuristic Bypass
        action_verbs = [
            'open', 'run', 'start', 'click', 'type', 'search', 'delete',
            'create', 'make', 'do', 'close', 'kill'
        ]
        first_word = user_text.strip().split()[0].lower() if user_text.strip() else ""
        if first_word in action_verbs:
            logger.debug("Deterministic heuristic bypass matched verb: %s", first_word)
            return "task", None

        system_prompt = """
        You are a routing agent for a local OS controller.
        Classify the user's input as either 'question' or 'task'.

        - 'task': The user is asking you to perform a physical action on the computer, like opening an app, running a command, clicking, or managing files.
        - 'question': The user is just chatting, greeting, or asking for information.

        OUTPUT FORMAT:
        You must output ONLY valid JSON in this exact format.
        {
            "intent": "task" | "question"
        }
        """

        try:
            logger.debug("Routing intent via Ollama using model %s...", self.model_name)
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                format='json'
            )

            content = response.get('message', {}).get('content', '').strip()

            try:
                result = json.loads(content)
                intent = result.get('intent', '').lower()
            except json.JSONDecodeError:
                # Ultimate fallback
                intent = content.lower()

            # If the model explicitly says task, we treat it as a task.
            if "task" in intent:
                return "task", None

            # Otherwise, default to question/chat behavior
            logger.debug("Intent classified as question. Generating answer...")
            ans_response = await asyncio.to_thread(
                ollama.chat,
                model=self.model_name,
                messages=[{"role": "user", "content": user_text}]
            )
            return "question", ans_response.get('message', {}).get('content', '')

        except Exception as e: # pylint: disable=broad-exception-caught
            logger.error("Router failed to classify intent: %s", e)
            raise RuntimeError(f"Router error: {e}") from e
