"""
Automated Test for evaluating Hierarchical Framework stability and hardware use.
"""
import asyncio
import psutil
from rich.console import Console

from core.config import config
from core.critic import Critic
from core.executor import Executor
from core.orchestrator import Orchestrator
from core.planner import Planner
from core.router import Router
from memory.chroma_store import ChromaStore
from vision.ui_parser import UIParser

console = Console()

def count_brave_processes():
    count = 0
    for proc in psutil.process_iter(['name']):
        try:
            if proc.info['name'] and 'brave.exe' in proc.info['name'].lower():
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return count

async def run_test():
    console.print("[bold cyan]Initializing Hybrid Framework Test...[/bold cyan]")
    
    config.auto_continue = True # Prevent blocking on CLI prompts during automated tests
    
    manager_model = config.manager_model
    if manager_model.startswith("gemini") and not config.gemini_api_key:
        console.print("[yellow]GEMINI_API_KEY not found. Falling back to local model for Manager...[/yellow]")
        manager_model = "llama3:latest"
        
    router = Router(model_name=config.worker_model)
    planner = Planner(model_name=manager_model)
    critic = Critic(model_name=manager_model)
    ui_parser = UIParser()
    executor = Executor(ui_parser)
    memory = ChromaStore()

    orchestrator = Orchestrator(
        router=router,
        planner=planner,
        executor=executor,
        memory=memory,
        critic=critic,
        vision_model=config.vision_model
    )

    results = []

    for i in range(1, 11):
        console.print(f"\n[bold magenta]=== Test Iteration {i}/10 ===[/bold magenta]")
        
        pre_count = count_brave_processes()
        console.print(f"Brave processes before: [yellow]{pre_count}[/yellow]")
        
        prompt = f"Open BRAVE and then open GOOGLE GEMINI AI. in the same brave tab. (Test Iteration {i})"
        
        try:
            await orchestrator.process_prompt(prompt)
            success = True
        except Exception as e:
            console.print(f"[red]Error during execution: {e}[/red]")
            success = False
            
        post_count = count_brave_processes()
        console.print(f"Brave processes after: [yellow]{post_count}[/yellow]")
        
        diff = post_count - pre_count
        if diff > 5: # Brave uses multiple processes per tab. A jump > 5 might mean a new window.
            console.print("[bold red]WARNING: Excessive Brave processes spawned! Possible runaway loop.[/bold red]")
            
        results.append({
            "iteration": i,
            "pre_count": pre_count,
            "post_count": post_count,
            "diff": diff,
            "success": success
        })
        
    # Clean up
    await executor.shutdown()
    
    console.print("\n[bold green]=== Test Summary ===[/bold green]")
    for r in results:
        status_color = "green" if r['success'] and r['diff'] < 10 else "red"
        console.print(f"Iter {r['iteration']}: Diff = [{status_color}]{r['diff']}[/{status_color}], Success = [{status_color}]{r['success']}[/{status_color}]")

if __name__ == "__main__":
    asyncio.run(run_test())
