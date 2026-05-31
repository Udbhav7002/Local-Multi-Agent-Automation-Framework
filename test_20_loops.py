import asyncio
from core.orchestrator import Orchestrator
from core.router import Router
from core.planner import Planner
from core.critic import Critic
from core.executor import Executor
from memory.chroma_store import ChromaStore
from vision.ui_parser import UIParser
import time


async def run_20_loops():
    print("Initializing Agent for 20 loops test...")
    router = Router(model_name="llama3.2:1b")
    planner = Planner(model_name="llama3.1:8b")
    critic = Critic(model_name="llama3.1:8b")
    executor = Executor(ui_parser=UIParser())
    memory = ChromaStore()
    orchestrator = Orchestrator(router, planner, executor, memory, critic=critic)

    for i in range(1, 21):
        print(f"\n{'=' * 40}\n[TEST] Starting Loop {i} of 20\n{'=' * 40}")

        try:
            await orchestrator.process_prompt("open brave")
        except Exception as e:
            print(f"[TEST ERROR] Loop {i} failed: {e}")

        print(
            f"[TEST] Finished Loop {i}. Waiting 5 seconds before next loop...")
        await asyncio.sleep(5)

    print("\n[TEST] Successfully completed 20 test loops.")

if __name__ == "__main__":
    asyncio.run(run_20_loops())
