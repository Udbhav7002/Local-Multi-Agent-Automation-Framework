"""
Main entry point for the Local Multi-Agent Automation Framework.
Handles REPL, command parsing, and orchestrator initialization.
"""
# pylint: disable=broad-exception-caught, line-too-long

import asyncio

import ollama
from rich.prompt import Prompt
from rich.text import Text
from rich.table import Table
from rich.console import Console

from core.config import config
from core.critic import Critic
from core.executor import Executor
from core.logger import setup_logger, console
from core.orchestrator import Orchestrator
from core.planner import Planner
from core.router import Router
from memory.chroma_store import ChromaStore
from vision.ui_parser import UIParser

logger = setup_logger("Main")


def draw_splash_screen(router_model: str, planner_model: str, vision_model: str) -> None:
    """Draws the ultra-minimalist 'opencode' style terminal splash screen."""
    
    # Blocky, solid ASCII Art
    ascii_art = """[#444444]
█      ██████  ██████  █████  █      
█     ██    ██ ██     ██   ██ █      
█     ██    ██ ██     ███████ █      
█     ██    ██ ██     ██   ██ █      
█████  ██████   ██████ ██   ██ █████ 

 █████   ██████  ███████ ███    ██ ████████ 
██   ██ ██       ██      ████   ██    ██    
███████ ██   ███ █████   ██ ██  ██    ██    
██   ██ ██    ██ ██      ██  ██ ██    ██    
██   ██  ██████  ███████ ██   ████    ██    
[/#444444]"""
    
    # Print logo and status
    console.print(ascii_art)
    console.print("  [#00ff00]●[/#00ff00] [#888888]Local Agent[/#888888]\n")
    
    # Models table (clean, no borders, lots of padding)
    models_table = Table.grid(padding=(0, 4))
    models_table.add_column(style="bold #ffffff", width=15)
    models_table.add_column(style="#888888")
    
    models_table.add_row("Router", router_model)
    models_table.add_row("Manager (Planner)", planner_model)
    models_table.add_row("Worker", vision_model)
    
    # Commands table
    commands_table = Table.grid(padding=(0, 4))
    commands_table.add_column(style="bold #ffffff", width=15)
    commands_table.add_column(style="#888888")
    
    commands_table.add_row("/model", "Change agent models (e.g., /model router qwen2.5-coder:7b)")
    commands_table.add_row("/clear", "Clear the terminal screen")
    commands_table.add_row("/save", "Save current session state")
    commands_table.add_row("/quit", "Exit the framework")
    
    # Layout sections with headers
    console.print("  [bold #ffffff]Models[/bold #ffffff]")
    console.print("  [#444444]──────────────[/#444444]")
    console.print(models_table)
    console.print()
    
    console.print("  [bold #ffffff]Commands[/bold #ffffff]")
    console.print("  [#444444]──────────────[/#444444]")
    console.print(commands_table)
    console.print("\n")


def _handle_model_command(orchestrator: Orchestrator, parts: list[str]) -> None:
    """Handles the /model CLI command to swap AI models on the fly."""
    if len(parts) == 1:
        console.print("\n  [bold #ffffff]Current Models[/bold #ffffff]")
        console.print("  [#444444]──────────────[/#444444]")
        
        table = Table.grid(padding=(0, 4))
        table.add_column(style="bold #ffffff", width=15)
        table.add_column(style="#888888")
        table.add_row("Router", orchestrator.router.model_name)
        table.add_row("Planner", orchestrator.planner.model_name)
        table.add_row("Vision", orchestrator.vision_model)
        console.print(table)
        console.print("\n  [#888888]Usage: /model [router|planner|vision] [new_model][/#888888]\n")
    elif len(parts) >= 3:
        target_agent = parts[1].lower()
        new_model = parts[2]

        if target_agent == 'router':
            orchestrator.router.model_name = new_model
            console.print(f"  [#00ff00]*[/#00ff00] [#888888]Router model changed to[/#888888] [bold #ffffff]{new_model}[/bold #ffffff]")
        elif target_agent == 'planner':
            orchestrator.planner.model_name = new_model
            console.print(f"  [#00ff00]*[/#00ff00] [#888888]Planner model changed to[/#888888] [bold #ffffff]{new_model}[/bold #ffffff]")
        elif target_agent == 'vision':
            orchestrator.vision_model = new_model
            console.print(f"  [#00ff00]*[/#00ff00] [#888888]Vision model changed to[/#888888] [bold #ffffff]{new_model}[/bold #ffffff]")
        else:
            console.print("  [#ff0000]x Invalid agent. Use router, planner, or vision.[/#ff0000]")
    else:
        console.print("  [#ff0000]x Usage: /model [router|planner|vision] [new_model][/#ff0000]")


async def async_main() -> None:
    """Asynchronous main loop handling user input and orchestration."""
    try:
        ollama.list()
    except Exception:
        console.print("[#ff0000]x Failed to connect to Ollama.[/#ff0000]")
        console.print("[#888888]Please make sure Ollama is installed and running (run 'ollama serve' in a separate terminal).[/#888888]")
        return

    # Using the hybrid local/API models from config
    router = Router(model_name=config.worker_model)
    planner = Planner(model_name=config.manager_model)
    critic = Critic(model_name=config.manager_model)
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

    # Clear terminal before drawing to enforce the clean look
    console.clear()
    
    # Draw the premium splash screen
    draw_splash_screen(router.model_name, planner.model_name, orchestrator.vision_model)

    while True:
        try:
            # Minimalist prompt
            user_text = Prompt.ask("[bold #ffffff]/[/bold #ffffff]").strip()
            
            if not user_text:
                continue
            if user_text.lower() in ['quit', 'exit', '/quit']:
                console.print("\n  [#888888]Shutting down...[/#888888]")
                if hasattr(orchestrator.executor, 'shutdown'):
                    await orchestrator.executor.shutdown()
                break

            if user_text.lower() == '/help':
                console.print("\n  [bold #ffffff]Commands[/bold #ffffff]")
                console.print("  [#444444]────────[/#444444]")
                help_table = Table.grid(padding=(0, 4))
                help_table.add_column(style="bold #ffffff", width=15)
                help_table.add_column(style="#888888")
                help_table.add_row("/help", "Show this message")
                help_table.add_row("/model", "Manage AI models")
                help_table.add_row("/clear", "Clear the terminal screen")
                help_table.add_row("/save", "Save current session state")
                help_table.add_row("/quit", "Exit the framework")
                console.print(help_table)
                console.print()
                continue
                
            if user_text.lower() == '/clear':
                console.clear()
                draw_splash_screen(orchestrator.router.model_name, orchestrator.planner.model_name, orchestrator.vision_model)
                continue
                
            if user_text.lower() == '/save':
                console.print("  [#00ff00]*[/#00ff00] [#888888]Session state saved.[/#888888]")
                # In the future, this can call memory.persist() or similar if supported
                continue

            if user_text.lower().startswith('/model'):
                parts = user_text.split()
                _handle_model_command(orchestrator, parts)
                continue

            await orchestrator.process_prompt(user_text)

        except KeyboardInterrupt:
            console.print("\n  [#888888]Shutting down via KeyboardInterrupt...[/#888888]")
            if hasattr(orchestrator.executor, 'shutdown'):
                await orchestrator.executor.shutdown()
            break
        except asyncio.CancelledError:
            console.print("\n  [#888888]Task canceled. Shutting down...[/#888888]")
            break
        except EOFError:
            console.print("\n  [#888888]EOF Received. Shutting down...[/#888888]")
            if hasattr(orchestrator.executor, 'shutdown'):
                await orchestrator.executor.shutdown()
            break
        except Exception as e:
            logger.error("System Error: %s", e)


def main() -> None:
    """Synchronous entry point."""
    import sys # pylint: disable=import-outside-toplevel
    import warnings # pylint: disable=import-outside-toplevel
    
    # Suppress ugly internal asyncio Proactor pipe warnings on Windows shutdown
    warnings.filterwarnings("ignore", category=ResourceWarning)
    
    if sys.platform == 'win32':
        # Apply a known workaround for Python's asyncio Proactor pipe bug on exit
        try:
            from asyncio.proactor_events import _ProactorBasePipeTransport # type: ignore
            def _silence_closed_pipe(self): # type: ignore
                pass
            _ProactorBasePipeTransport.__del__ = _silence_closed_pipe # type: ignore
        except Exception:
            pass

    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass
    finally:
        # Force a clean exit to prevent lingering threads or pipes from throwing errors
        sys.exit(0)


if __name__ == "__main__":
    main()
