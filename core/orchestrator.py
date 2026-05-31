"""
Orchestrator module for the Local Multi-Agent Automation Framework.
Handles the core execution loop, routing, planning, and memory reflection.
"""
# pylint: disable=broad-exception-caught, too-many-arguments, too-many-positional-arguments, line-too-long

import asyncio
import json
from typing import Any, Dict, List, Tuple

from rich.panel import Panel

from core.config import config
from core.critic import Critic
from core.executor import Executor
from core.logger import setup_logger, console
from core.planner import Planner
from core.router import Router
from memory.chroma_store import ChromaStore
from vision.verifier import VisionVerifier

logger = setup_logger("Orchestrator")


class Orchestrator: # pylint: disable=too-few-public-methods
    """
    The central hub connecting the Router, Planner, Critic, Executor, and Memory.
    """
    def __init__(
            self,
            router: Router,
            planner: Planner,
            executor: Executor,
            memory: ChromaStore,
            critic: Critic,
            vision_model: str = "llava") -> None:
        self.router = router
        self.planner = planner
        self.executor = executor
        self.memory = memory
        self.critic = critic
        self.vision_model = vision_model

    async def process_prompt(self, user_text: str) -> None:
        """
        The main execution loop for a user prompt.
        """
        logger.info("Processing user input: '%s'", user_text)

        # Step 1: Memory Check
        logger.debug("Checking memory cache...")
        cached_plan = self.memory.get_plan(user_text)
        if cached_plan:
            logger.info(
                "[bold green]Found successful plan in memory! Bypassing planner...[/bold green]")
            await self._execute_plan(user_text, cached_plan)
            return

        # Step 2: Routing
        logger.info("Routing intent...")
        try:
            intent, response = await self.router.classify(user_text)
        except Exception as e:
            logger.error("Routing failed: %s", e)
            return

        if intent == "question":
            console.print(Panel(response, title="Agent Answer", border_style="blue"))
            return

        logger.info("[bold cyan]Identified as a TASK. Planning execution...[/bold cyan]")

        # Step 3 & 4: Planning & Execution Loop
        await self._plan_and_execute(user_text)

    async def _plan_and_execute(self, user_text: str) -> None:
        """Handles the planning and execution loop with retries and reflection memory."""
        current_prompt_context = user_text

        # Check Reflection Memory for past failures
        past_failures = self.memory.get_failures(user_text)
        if past_failures:
            failure_context = "\n\n[REFLECTION MEMORY] Warning: The following strategies failed previously for this or a similar task. DO NOT REPEAT THEM.\n"
            for failure in past_failures:
                failure_context += f"- Failed Plan: {failure['plan']}\n  Reason: {failure['reason']}\n"
            current_prompt_context += failure_context
            logger.warning(
                "[yellow]Loaded %s past failure(s) into Planner context.[/yellow]", len(past_failures))

        for plan_attempt in range(config.max_plan_regenerations):
            if plan_attempt > 0:
                logger.info(
                    "[magenta]Requesting new plan from Planner (Attempt %s/%s)...[/magenta]",
                    plan_attempt + 1, config.max_plan_regenerations)
            else:
                logger.info("Generating step-by-step plan...")

            try:
                plan_json = await self.planner.generate_plan(current_prompt_context)
                steps = json.loads(plan_json)
                if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
                    raise ValueError(
                        "Generated plan is not a JSON array of strings.")
                
                # Pretty print plan
                plan_text = ""
                for idx, step in enumerate(steps):
                    plan_text += f"[bold]{idx + 1}.[/bold] [cyan]Goal[/cyan] -> {step}\n"
                console.print(Panel(plan_text.strip(), title=f"Execution Plan ({len(steps)} steps)", border_style="magenta"))

            except (json.JSONDecodeError, ValueError) as e:
                logger.error("Error parsing plan from planner: %s", e)
                current_prompt_context += f"\n\n[SYSTEM FEEDBACK]: Your previous response was not valid JSON. Error: {e}.\nCRITICAL INSTRUCTION: You MUST output ONLY a valid JSON array."
                continue
            except Exception as e:
                logger.error("Planner exception: %s", e)
                current_prompt_context += f"\n\n[SYSTEM FEEDBACK]: Your previous generation failed with error: {e}.\nCRITICAL INSTRUCTION: Fix the error and generate a valid JSON plan."
                continue

            # Multi-Agent Concurrency: Critic Evaluation
            logger.info("Passing plan to Critic for evaluation...")
            is_approved, critic_feedback = await self.critic.verify_plan(user_text, plan_json)

            if not is_approved:
                logger.warning("[red]Critic rejected the plan:[/red] %s. Regenerating...", critic_feedback)
                current_prompt_context += f"\n\n[SYSTEM FEEDBACK]: Your previous plan was rejected by the Critic Agent. Reason: {critic_feedback}\nCRITICAL INSTRUCTION: You MUST fix these issues and generate a new plan."
                continue
            
            logger.info("[green]Critic approved the plan.[/green]")

            # Execute the plan
            plan_success, failure_feedback = await self._execute_steps(steps)

            if plan_success:
                logger.info("[bold green]Task Complete.[/bold green]")
                self.memory.save_plan(user_text, plan_json)
                return

            if failure_feedback:
                logger.warning(
                    "Plan failed. Passing feedback to planner for a new strategy...")
                # Save to Reflection Memory
                self.memory.save_failure(
                    user_text, plan_json, failure_feedback.strip())
                current_prompt_context += failure_feedback
                continue

            # Failed but no feedback (e.g., user aborted or permanent
            # non-recoverable failure)
            self.memory.save_failure(
                user_text, plan_json, "User aborted or catastrophic error")
            return

        logger.error(
            "[bold red]Task failed completely after maximum plan regenerations.[/bold red]")

    async def _execute_plan(
            self,
            user_text: str,
            plan_json: str) -> None:
        """Executes a pre-existing plan (e.g. from cache)."""
        try:
            steps = json.loads(plan_json)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse cached plan: %s", e)
            return

        success, _ = await self._execute_steps(steps)
        if success:
            logger.info("[bold green]Task Complete (from Cache).[/bold green]")
        else:
            logger.warning("[bold red]Cached plan execution failed.[/bold red]")
            self.memory.save_failure(
                user_text, plan_json, "Cached plan failed upon execution.")

    async def _execute_steps( # pylint: disable=too-many-locals, too-many-branches, too-many-statements, too-many-nested-blocks
            self, steps: List[str]) -> Tuple[bool, str]:
        """
        Executes a list of high-level sub-tasks via the Worker Agent.
        Returns (success_boolean, failure_feedback_string).
        """
        from core.worker import Worker
        worker = Worker(config.worker_model)
        action_history = ""

        for i, sub_task in enumerate(steps):
            logger.info("Manager delegating sub-task %s to Worker: '%s'", i + 1, sub_task)
            
            try:
                worker_json = await worker.generate_action(sub_task, action_history)
                worker_steps = json.loads(worker_json)
            except Exception as e:
                logger.error("Worker failed to break down task: %s", e)
                return False, f"Worker Agent failed to generate actions for sub-task '{sub_task}': {e}"
                
            for j, step in enumerate(worker_steps):
                action = step.get('action', 'unknown')
                target = step.get('target', 'unknown')
                method = step.get('method', 'cli')
                expected = step.get('expected_outcome', 'unknown')

                step_success = False
                for attempt in range(config.max_step_retries + 1):
                    if attempt > 0:
                        logger.info(
                            "Retrying Worker Step %s.%s (Attempt %s/%s)...",
                            i + 1, j + 1, attempt + 1, config.max_step_retries + 1)
                    else:
                        logger.info(
                            "Worker Step %s.%s: [cyan]%s[/cyan] %s (Method: %s)", i + 1, j + 1, action, target, method)

                    if method == "cli" and not self._is_safe_command(target):
                        logger.warning("[bold yellow]Dangerous action detected.[/bold yellow]")
                        ans = input(f"Run CLI command: `{target}`? [y/N]: ")
                        if ans.strip().lower() != 'y':
                            logger.warning(
                                "Action canceled by user. Aborting task.")
                            return False, ""

                    success, output = await self.executor.execute_step(step)
                    action_history += f"Action: {action} | Target: {target} | Method: {method} | Success: {success} | Output: {output}\n"

                    if success:
                        logger.info("[green]Worker Step %s.%s completed:[/green] %s", i + 1, j + 1, output)

                        # Vision Verification if applicable
                        if expected and expected != "unknown":
                            logger.info("Waiting for UI to render before taking screenshot (4s)...")
                            await asyncio.sleep(4)
                            v_success, v_msg = await self._verify_vision(expected)

                            if v_success:
                                logger.info("[green]Vision Verification passed:[/green] %s", v_msg)
                                step_success = True
                                break

                            logger.warning("[yellow]Vision Verification failed:[/yellow] %s", v_msg)
                            if attempt < config.max_step_retries:
                                logger.info("Auto-restarting step...")
                                await asyncio.sleep(2)
                                continue

                            if config.auto_continue:
                                logger.warning(
                                    "Vision failed, but auto-continuing due to config.")
                                step_success = True
                                break

                            feedback = f"\n\n[SYSTEM FEEDBACK]: Worker Step {i + 1}.{j + 1} (`{action}` `{target}`) failed vision verification.\nVision Model Feedback: {v_msg}\nCRITICAL INSTRUCTION: DO NOT blindly start from scratch. Analyze the Vision Model Feedback to understand the current screen state."
                            return False, feedback

                        # Expected outcome not defined, so step succeeds automatically
                        step_success = True
                        break

                    # If success was false
                    logger.error("[red]Worker Step %s.%s failed:[/red] %s", i + 1, j + 1, output)

                    if attempt < config.max_step_retries:
                        logger.info("Auto-restarting step...")
                        await asyncio.sleep(2)
                        continue

                    feedback = f"\n\n[SYSTEM FEEDBACK]: Worker Step {i + 1}.{j + 1} (`{action}` `{target}`) failed with error: {output}.\nCRITICAL INSTRUCTION: DO NOT blindly start from scratch. Analyze the failure to understand the current state."
                    return False, feedback

                if not step_success:
                    return False, ""

        return True, ""

    async def _verify_vision(self, expected_outcome: str) -> Tuple[bool, str]:
        """Runs the vision verification logic."""
        try:
            verifier = VisionVerifier(model_name=self.vision_model)
            return await verifier.verify(expected_outcome)
        except Exception as e:
            logger.error("Vision verification error: %s", e)
            return False, f"Error: {e}"

    def _is_safe_command(self, cmd: str) -> bool:
        """Heuristics to determine if a command needs confirmation."""
        cmd_lower = cmd.lower()
        dangerous_keywords = [
            'rm ',
            'del ',
            'format',
            'mkfs',
            'shutdown',
            'reboot',
            'kill']
        return not any(keyword in cmd_lower for keyword in dangerous_keywords)
