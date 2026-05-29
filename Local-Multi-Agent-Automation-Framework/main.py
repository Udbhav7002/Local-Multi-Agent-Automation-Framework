import asyncio
import sys
from core.orchestrator import Orchestrator
from core.router import Router
from core.planner import Planner
from core.executor import Executor
from memory.chroma_store import ChromaStore
from vision.ui_parser import UIParser

import ollama

async def async_main():
    print("[ORCHESTRATOR] Initializing multi-agent automation system...")
    
    try:
        ollama.list()
    except Exception:
        print("[ERROR] Failed to connect to Ollama.")
        print("[ERROR] Please make sure Ollama is installed and running (run 'ollama serve' in a separate terminal).")
        return

    # Using the local installed model
    router = Router(model_name="llama3:latest")
    planner = Planner(model_name="llama3:latest")
    
    # Init other components
    executor = Executor(ui_parser_stub=UIParser())
    memory = ChromaStore()
    
    orchestrator = Orchestrator(router, planner, executor, memory, vision_model="llava:latest")
    
    print("\n[ORCHESTRATOR] System Ready. Type 'quit' or 'exit' to stop.")
    
    while True:
        try:
            user_text = input("❯ ").strip()
            if not user_text:
                continue
            if user_text.lower() in ['quit', 'exit']:
                print("[ORCHESTRATOR] Shutting down...")
                if hasattr(orchestrator.executor, 'shutdown'):
                    await orchestrator.executor.shutdown()
                break
                
            if user_text.strip().lower().startswith('/model'):
                parts = user_text.strip().split()
                if len(parts) == 1:
                    print(f"\n--- Current Models ---")
                    print(f"Router:  {orchestrator.router.model_name}")
                    print(f"Planner: {orchestrator.planner.model_name}")
                    print(f"Vision:  {orchestrator.vision_model}")
                    print(f"----------------------")
                    print("To change, type: /model [router|planner|vision] [new_model]\n")
                elif len(parts) >= 3:
                    target_agent = parts[1].lower()
                    new_model = parts[2]
                    
                    if target_agent == 'router':
                        orchestrator.router.model_name = new_model
                        print(f"[ORCHESTRATOR] Router model changed to {new_model}")
                    elif target_agent == 'planner':
                        orchestrator.planner.model_name = new_model
                        print(f"[ORCHESTRATOR] Planner model changed to {new_model}")
                    elif target_agent == 'vision':
                        orchestrator.vision_model = new_model
                        print(f"[ORCHESTRATOR] Vision model changed to {new_model}")
                    else:
                        print("[ERROR] Invalid agent. Use router, planner, or vision.")
                else:
                    print("[ERROR] Invalid format. Use: /model [router|planner|vision] [new_model]")
                continue
                
            await orchestrator.process_prompt(user_text)
        except KeyboardInterrupt:
            print("\n[ORCHESTRATOR] Shutting down...")
            if hasattr(orchestrator.executor, 'shutdown'):
                await orchestrator.executor.shutdown()
            break
        except asyncio.CancelledError:
            print("\n[ORCHESTRATOR] Task canceled. Shutting down...")
            break
        except EOFError:
            print("\n[ORCHESTRATOR] Shutting down...")
            if hasattr(orchestrator.executor, 'shutdown'):
                await orchestrator.executor.shutdown()
            break
        except Exception as e:
            err_msg = str(e) if str(e) else repr(e)
            print(f"[ERROR] System Error: {err_msg}")

def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
