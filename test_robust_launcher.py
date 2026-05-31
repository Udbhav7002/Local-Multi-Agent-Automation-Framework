import asyncio
import os
import random
import subprocess
import time
from unittest.mock import patch

from core.router import Router
from core.planner import Planner
from core.critic import Critic
from core.executor import Executor
from core.orchestrator import Orchestrator

# Simple in-memory stub for testing


class MemoryStub:
    def get_plan(self, user_text):
        return None

    def save_plan(self, user_text, plan):
        pass

    def get_failures(self, user_text):
        return []

    def save_failure(self, user_text, plan, reason):
        pass


APPS = {
    "Notepad": "notepad.exe",
    "Calculator": "calculator.exe",
    "Paint": "mspaint.exe",
    "Command Prompt": "cmd.exe",
    "File Explorer": "explorer.exe",
}


async def run_test():
    print("=========================================")
    print(" ROBUST LAUNCHER TEST SUITE (10 ITERATIONS) ")
    print("=========================================")

    router = Router(model_name="llama3.1:8b")
    planner = Planner(model_name="llama3.1:8b")
    critic = Critic(model_name="llama3.1:8b")
    executor = Executor()
    memory = MemoryStub()
    orchestrator = Orchestrator(
        router,
        planner,
        executor,
        memory,
        critic=critic,
        vision_model="llava:13b")

    # We want it to auto-fail/continue if Vision doesn't see it,
    # so we don't get stuck on a prompt. We'll set AUTO_CONTINUE.
    os.environ["AUTO_CONTINUE"] = "1"

    successes = 0
    total = 10

    app_names = list(APPS.keys())

    for i in range(total):
        app_name = random.choice(app_names)
        exe_name = APPS[app_name]

        print(f"\n--- Iteration {i + 1}/{total} ---")
        print(f"Target App: {app_name}")

        # We will patch 'print' so we can capture if vision model says Success:
        # True
        vision_success = False
        original_print = print

        def mock_print(*args, **kwargs):
            nonlocal vision_success
            text = " ".join(str(a) for a in args)
            original_print(*args, **kwargs)
            if "[VISION] Verifier response:" in text and "(Success: True)" in text:
                vision_success = True

        prompt = f"open {app_name}"

        with patch("builtins.print", mock_print):
            try:
                await orchestrator.process_prompt(prompt)
            except Exception as e:
                original_print(f"Error processing prompt: {e}")

        if vision_success:
            successes += 1
            original_print(f"[PASS] Iteration {i + 1} Passed!")
        else:
            original_print(f"[FAIL] Iteration {i + 1} Failed!")

        # Cleanup: close the app
        original_print(f"Cleaning up {exe_name}...")
        subprocess.run(
            f"taskkill /IM {exe_name} /F",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)

        # Special case for calculator on Windows 10/11 which is
        # CalculatorApp.exe
        if app_name == "Calculator":
            subprocess.run(
                "taskkill /IM CalculatorApp.exe /F",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)

        time.sleep(2)

    print("\n=========================================")
    print(f" TEST SUITE COMPLETE: {successes}/{total} SUCCESSFUL ")
    print("=========================================")

    if successes >= 8:
        print("[SUCCESS] GOAL ACHIEVED!")
    else:
        print("[WARNING] GOAL FAILED. Adjust execution logic and try again.")

if __name__ == "__main__":
    asyncio.run(run_test())
