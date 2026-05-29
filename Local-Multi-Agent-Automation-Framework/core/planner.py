import ollama
import asyncio
import json
import re

class Planner:
    def __init__(self, model_name="llama3.1:8b"):
        self.model_name = model_name

    async def generate_plan(self, task_description: str) -> str:
        """
        Takes a task and outputs a strict JSON array of steps.
        Enforces a CLI-first policy.
        """
        system_prompt = """
        You are the Planning Agent for a local OS controller.
        Your job is to break down the user's task into a strict JSON array of steps.
        
        RULES:
        1. YOU MUST PRIORITIZE CLI ACTIONS for system management or file operations.
        2. To OPEN LOCAL APPLICATIONS ANYWHERE ON THE COMPUTER, use the `system` method with action `open_app` and the name of the app (e.g. 'notepad'). Do not use `search_start_menu` or `run_command` for opening applications.
        3. For WEB TASKS (navigating websites, filling forms, searching the web), you MUST prioritize the "browser" method, which runs blazingly fast in the background.
           - Valid actions for "browser": "goto", "click_dom", "type_dom".
           - For "goto", the target should be a URL.
           - For "click_dom", the target should be a CSS selector.
           - For "type_dom", the target MUST be formatted as: CSS_SELECTOR:::TEXT_TO_TYPE
        4. If "method": "cli", the "target" should be the exact command string. USE WINDOWS POWERSHELL COMMANDS ONLY (no Linux commands like xdotool).
        5. If "method": "gui", the "action" can be "click", "type", "hotkey", or "search_start_menu". For "hotkey", target is keys separated by '+' (e.g. "ctrl+w", "alt+f4").
        6. If "method": "system", valid actions are "sleep" (target is seconds in numbers), "close_browser" (target is "browser"), and "open_app" (target is the app name).
        
        OUTPUT FORMAT:
        You must output ONLY valid JSON in this exact format. No markdown blocks, no other text.
        [
            {
                "action": "run_command" | "click" | "type" | "hotkey" | "goto" | "click_dom" | "type_dom" | "sleep" | "close_browser" | "open_app",
                "target": "string (command, element name, URL, CSS selector, or seconds)",
                "method": "cli" | "gui" | "browser" | "system",
                "expected_outcome": "string describing what success looks like"
            }
        ]
        
        EXAMPLE FOR WEB TASKS:
        [
            { "action": "goto", "target": "https://gemini.google.com/", "method": "browser", "expected_outcome": "Gemini website is open" },
            { "action": "sleep", "target": "10", "method": "system", "expected_outcome": "Waited 10 seconds" },
            { "action": "close_browser", "target": "browser", "method": "system", "expected_outcome": "Browser closed" }
        ]
        
        EXAMPLE FOR OPENING APPS:
        [
            { "action": "open_app", "target": "brave", "method": "system", "expected_outcome": "Brave browser opened" }
        ]
        """
        
        try:
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task_description}
                ]
            )
            
            content = response['message']['content'].strip()
            
            # Remove <think>...</think> blocks for reasoning models like DeepSeek-R1
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            
            # Extract JSON array robustly using non-greedy regex
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                content = match.group(0)
            
            # Simple validation to ensure it's JSON
            json.loads(content)
                
            return content.strip()
            
        except Exception as e:
            # Re-raise the exception so the orchestrator can handle it
            raise e
