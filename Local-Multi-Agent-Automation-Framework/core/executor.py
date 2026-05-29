import asyncio
import subprocess
import traceback

class Executor:
    def __init__(self, ui_parser_stub=None):
        self.ui_parser = ui_parser_stub
        self._pw = None
        self._browser = None
        self._page = None

    def _get_browser_path(self, browser_name: str):
        """Dynamically finds the browser executable path via Windows Registry."""
        import winreg
        for hive in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
            try:
                key = winreg.OpenKey(hive, rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{browser_name}.exe")
                path, _ = winreg.QueryValueEx(key, "")
                return path
            except Exception:
                continue
        return None

    async def _init_browser(self):
        if not self._pw:
            from playwright.async_api import async_playwright
            self._pw = await async_playwright().start()
            import os
            
            brave_path = self._get_browser_path("brave")
            if brave_path and os.path.exists(brave_path):
                print(f"[BROWSER] Launching Brave Browser natively...")
                self._browser = await self._pw.chromium.launch(headless=False, executable_path=brave_path)
            else:
                print(f"[BROWSER] Brave not found. Launching default Chromium browser...")
                self._browser = await self._pw.chromium.launch(headless=False)
            self._page = await self._browser.new_page()

    async def shutdown(self):
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()

    async def execute_step(self, step: dict) -> tuple[bool, str]:
        """
        Executes a single planned step.
        Returns (success: bool, output/message: str)
        """
        method = step.get('method')
        action = step.get('action')
        target = step.get('target')

        try:
            # Smart fallback for LLM hallucination: 
            # Force the correct execution pipeline based on the action, regardless of what 'method' the LLM guessed.
            if action == "run_command":
                method = "cli"
            elif action in ["goto", "click_dom", "type_dom"]:
                method = "browser"
            elif action in ["click", "type", "hotkey", "search_start_menu"]:
                method = "gui"
            elif action in ["sleep", "close_browser", "open_app"]:
                method = "system"

            if method == "cli":
                return await self._execute_cli(target)
            elif method == "gui":
                return await self._execute_gui(action, target)
            elif method == "browser":
                return await self._execute_browser(action, target)
            elif method == "system":
                if action == "sleep":
                    print(f"[SYSTEM] Sleeping for {target} seconds...")
                    await asyncio.sleep(float(target))
                    return True, f"Slept for {target} seconds."
                elif action == "close_browser":
                    print("[SYSTEM] Closing browser...")
                    await self.shutdown()
                    return True, "Closed the browser."
                elif action == "open_app":
                    print(f"[SYSTEM] Opening application: {target}...")
                    return await self._execute_cli(f"Start-Process {target}")
            else:
                return False, f"Unknown execution method: {method}"
        except Exception as e:
            err_msg = traceback.format_exc()
            return False, f"Exception during execution:\n{err_msg}"

    async def _execute_cli(self, command: str) -> tuple[bool, str]:
        """Executes a native command via subprocess."""
        # Using powershell for Windows compatibility by default, but this can be adjusted
        try:
            process = await asyncio.create_subprocess_shell(
                f"powershell -Command \"{command}\"",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            output = stdout.decode().strip()
            error_output = stderr.decode().strip()
            
            if process.returncode == 0:
                return True, output if output else "Command executed successfully with no output."
            else:
                return False, f"Command failed with code {process.returncode}: {error_output}"
                
        except Exception as e:
            return False, f"Failed to launch subprocess: {str(e)}"

    async def _execute_gui(self, action: str, target: str) -> tuple[bool, str]:
        """
        Executes a GUI action.
        This relies on the UI Parser to find coordinates, then pyautogui to act.
        """
        import pyautogui
        pyautogui.FAILSAFE = True
        
        if action == "type":
            print(f"[VISION] Typing '{target}'...")
            pyautogui.write(target, interval=0.05)
            pyautogui.press('enter')
            return True, f"Typed '{target}' and pressed enter."
            
        if action == "hotkey":
            print(f"[VISION] Pressing hotkey '{target}'...")
            keys = target.split('+')
            pyautogui.hotkey(*keys)
            return True, f"Pressed hotkey {target}."
            
        if action == "search_start_menu":
            print(f"[VISION] Opening Start Menu and searching for '{target}'...")
            pyautogui.press('win')
            await asyncio.sleep(1.0)
            pyautogui.write(target, interval=0.05)
            await asyncio.sleep(1.0)
            pyautogui.press('enter')
            return True, f"Searched Start Menu for '{target}' and pressed enter."
            
        coords = self.ui_parser.find_element(target)
        if not coords:
            # Fallback to general system commands if we can't find it visually
            if action == "click" and "start" in target.lower():
                pyautogui.press('win')
                return True, "Pressed Windows key as fallback for Start button."
            return False, f"Could not find element '{target}' on screen."
            
        x, y = coords
        
        if action == "click":
            print(f"[VISION] Moving mouse to ({x}, {y}) and clicking...")
            pyautogui.moveTo(x, y, duration=0.5)
            pyautogui.click()
            return True, f"Clicked on '{target}' at ({x}, {y})."
        else:
            return False, f"Unsupported GUI action: {action}"

    async def _execute_browser(self, action: str, target: str, retry=True) -> tuple[bool, str]:
        """
        Executes a browser action directly in the DOM using Playwright.
        """
        try:
            await self._init_browser()
            
            if action == "goto":
                print(f"[BROWSER] Navigating to {target}...")
                await self._page.goto(target, wait_until="load")
                return True, f"Navigated to {target}."
            elif action == "click_dom":
                print(f"[BROWSER] Clicking on selector: '{target}'...")
                await self._page.click(target)
                return True, f"Clicked element matching '{target}'."
            elif action == "type_dom":
                if ":::" not in target:
                    return False, "Target for type_dom must be in format 'selector:::text'."
                selector, text = target.split(":::", 1)
                print(f"[BROWSER] Typing '{text}' into selector: '{selector}'...")
                await self._page.fill(selector, text)
                return True, f"Typed text into '{selector}'."
            else:
                return False, f"Unsupported browser action: {action}"
        except Exception as e:
            err_str = str(e)
            if retry and ("Connection closed" in err_str or "Target closed" in err_str):
                print("[BROWSER] Connection dead. Restarting Playwright instance...")
                await self.shutdown()
                self._pw = None
                self._browser = None
                self._page = None
                return await self._execute_browser(action, target, retry=False)
            return False, f"Browser execution failed: {err_str}"
