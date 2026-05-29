import asyncio
from core.orchestrator import Orchestrator
from core.router import Router
from core.planner import Planner
from core.executor import Executor
from memory.chroma_store import ChromaStore
from vision.ui_parser import UIParser

import os

async def test():
    print("[TEST] Initializing test system...")
    # Use the models the user has chosen
    os.environ['AUTO_CONTINUE'] = '1'
    router = Router(model_name="llama3:latest")
    planner = Planner(model_name="llama3:latest")
    executor = Executor(ui_parser_stub=UIParser())
    memory = ChromaStore()
    
    orchestrator = Orchestrator(router, planner, executor, memory, vision_model="llava:latest")
    
    prompt = "Open calculator"
    print(f"[TEST] Sending prompt: {prompt}\n")
    await orchestrator.process_prompt(prompt)
    print("[TEST] Done.")

if __name__ == "__main__":
    asyncio.run(test())
