import asyncio
import json
import os

class Orchestrator:
    def __init__(self, router, planner, executor, memory, vision_model="llava"):
        self.router = router
        self.planner = planner
        self.executor = executor
        self.memory = memory
        self.vision_model = vision_model

    async def process_prompt(self, user_text: str) -> None:
        """
        The main execution loop for a user prompt.
        1. Check memory for exact match plan
        2. Route (Question vs Task)
        3. If Question -> Answer and return
        4. If Task -> Plan -> Execute steps -> Verify
        """
        print(f"\n[ORCHESTRATOR] Processing user input: '{user_text}'")

        # Step 1: Memory Check
        print("[ORCHESTRATOR] Checking memory cache...")
        cached_plan = self.memory.get_plan(user_text)
        if cached_plan:
            print("[ORCHESTRATOR] Found successful plan in memory! Bypassing planner...")
            plan_json = cached_plan
            intent = "task"
        else:
            # Step 2: Routing
            print("[ORCHESTRATOR] Routing intent...")
            print(f"[REASONING] Routing intent for prompt: '{user_text}'")
            intent, response = await self.router.classify(user_text)
            
            if intent == "question":
                print(f"\n[AGENT]\n{response}\n")
                return
                
            print("[ORCHESTRATOR] Identified as a TASK. Planning execution...")

            MAX_PLAN_REGENERATIONS = 3
            current_prompt_context = user_text
            
            for plan_attempt in range(MAX_PLAN_REGENERATIONS):
                if plan_attempt > 0:
                    print(f"\n[ORCHESTRATOR] Requesting new plan from Planner (Attempt {plan_attempt+1}/{MAX_PLAN_REGENERATIONS})...")
                else:
                    # Step 3: Planning
                    print("[ORCHESTRATOR] Generating step-by-step plan...")
                    
                try:
                    plan_json = await self.planner.generate_plan(current_prompt_context)
                    steps = json.loads(plan_json)
                    if not isinstance(steps, list) or not all(isinstance(s, dict) for s in steps):
                        raise ValueError("Generated plan is not a JSON array of objects.")
                    print(f"[ORCHESTRATOR] Plan generated with {len(steps)} steps.")
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Error parsing plan from planner: {str(e)}")
                    print("[ORCHESTRATOR] Invalid JSON generated. Passing failure back to planner...")
                    current_prompt_context += f"\n\n[SYSTEM FEEDBACK]: Your previous response was not valid JSON. Error: {str(e)}.\nCRITICAL INSTRUCTION: You MUST output ONLY a valid JSON array. No markdown, no conversational text."
                    continue # Skip execution, go straight to next plan attempt
                except Exception as e:
                    print(f"[ERROR] Planner exception: {str(e)}")
                    print("[ORCHESTRATOR] Planner failed. Passing failure back to planner...")
                    current_prompt_context += f"\n\n[SYSTEM FEEDBACK]: Your previous generation failed with error: {str(e)}.\nCRITICAL INSTRUCTION: Fix the error and generate a valid JSON plan."
                    continue

                # Step 4: Execution Loop
                plan_failed_permanently = False
                for i, step in enumerate(steps):
                    action = step.get('action', 'unknown')
                    target = step.get('target', 'unknown')
                    method = step.get('method', 'cli')
                    expected = step.get('expected_outcome', 'unknown')
                    
                    MAX_RETRIES = 1
                    step_success = False
                    for attempt in range(MAX_RETRIES + 1):
                        if attempt > 0:
                            print(f"\n[ORCHESTRATOR] Retrying Step {i+1} (Attempt {attempt+1}/{MAX_RETRIES+1})...")
                        else:
                            print(f"\n[ORCHESTRATOR] Step {i+1}: {action} {target} (Method: {method})")
                        
                        if method == "cli":
                            is_safe = await self._is_safe_command(target)
                            if not is_safe:
                                print(f"[ORCHESTRATOR] Action requires confirmation...")
                                ans = input(f"[WARNING] DANGEROUS ACTION DETECTED. Run CLI command: `{target}`? (y/n): ")
                                confirmed = ans.strip().lower() == 'y'
                                if not confirmed:
                                    print("[ORCHESTRATOR] Action canceled by user. Aborting task.")
                                    return
                            
                        success, output = await self.executor.execute_step(step)
                        
                        if success:
                            print(f"[ORCHESTRATOR] Step {i+1} completed. {output}")
                            if expected and expected != "unknown":
                                from vision.verifier import VisionVerifier
                                verifier = VisionVerifier(model_name=self.vision_model)
                                v_success, v_msg = await verifier.verify(expected)
                                if v_success:
                                    print(f"[VISION] Verification passed: {v_msg}")
                                    step_success = True
                                    break
                                else:
                                    print(f"[VISION] Verification failed: {v_msg}")
                                    if attempt < MAX_RETRIES:
                                        print("[ORCHESTRATOR] Auto-restarting step...")
                                        await asyncio.sleep(2)
                                        continue
                                    else:
                                        if "y" in os.environ.get("AUTO_CONTINUE", "").lower() or "1" in os.environ.get("AUTO_CONTINUE", ""):
                                            print("[ORCHESTRATOR] Vision model thinks the step failed after retries. Continue anyway? (y/n): y (Auto-continued)")
                                            ans = 'y'
                                        else:
                                            ans = input("[ORCHESTRATOR] Vision model thinks the step failed after retries. Continue anyway? (y/n): ")
                                        
                                        if ans.strip().lower() != 'y':
                                            print("[ORCHESTRATOR] Step failed permanently. Passing failure back to planner for a new strategy...")
                                            current_prompt_context += f"\n\n[SYSTEM FEEDBACK]: Your previous plan failed at Step {i+1} (`{action}` `{target}`) with error: Vision verification failed - {v_msg}.\nCRITICAL INSTRUCTION: You MUST generate a completely different alternative plan. DO NOT use the exact same `{action}` and `{target}`."
                                            plan_failed_permanently = True
                                            break
                                        else:
                                            print("[ORCHESTRATOR] Overriding vision model. Continuing task...")
                                            step_success = True
                                            break
                            else:
                                step_success = True
                                break
                        else:
                            print(f"[ERROR] Step {i+1} failed: {output}")
                            if attempt < MAX_RETRIES:
                                print("[ORCHESTRATOR] Auto-restarting step...")
                                await asyncio.sleep(2)
                                continue
                            else:
                                print("[ORCHESTRATOR] Step failed permanently. Passing failure back to planner for a new strategy...")
                                current_prompt_context += f"\n\n[SYSTEM FEEDBACK]: Your previous plan failed at Step {i+1} (`{action}` `{target}`) with error: {output}.\nCRITICAL INSTRUCTION: You MUST generate a completely different alternative plan. DO NOT use the exact same `{action}` and `{target}`."
                                plan_failed_permanently = True
                                break
                                
                    if plan_failed_permanently:
                        break
                        
                    if not step_success and not plan_failed_permanently:
                        return
                
                if not plan_failed_permanently:
                    print("\n[ORCHESTRATOR] Task Complete.")
                    self.memory.save_plan(user_text, plan_json)
                    return
            
            print("[ORCHESTRATOR] Task failed completely after maximum plan regenerations.")

    async def _is_safe_command(self, cmd: str) -> bool:
        """Heuristics to determine if a command needs confirmation."""
        cmd_lower = cmd.lower()
        dangerous_keywords = ['rm ', 'del ', 'format', 'mkfs', 'shutdown', 'reboot', 'kill']
        for keyword in dangerous_keywords:
            if keyword in cmd_lower:
                return False
        return True
