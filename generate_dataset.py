import json
import random

apps = {
    "msedge.exe": ["browse the web", "open edge", "surf the internet", "launch microsoft edge", "open web browser", "I need to look something up online"],
    "notepad.exe": ["open notepad", "take a note", "I need to write something down", "launch text editor", "open simple text editor"],
    "calc.exe": ["do some math", "open calculator", "launch calc", "I need to calculate something", "crunch some numbers", "open calc.exe"],
    "winword.exe": ["open word", "launch microsoft word", "write a document", "create a word doc", "I need to write a letter"],
    "excel.exe": ["open excel", "launch microsoft excel", "create a spreadsheet", "I need to do some accounting", "make a table"],
    "brave.exe": ["open brave", "launch brave browser", "surf privately", "open brave.exe"],
    "cmd.exe": ["open command prompt", "launch cmd", "open terminal", "I need to run some commands", "open command line"],
    "powershell.exe": ["open powershell", "launch powershell", "I need a better terminal", "powershell please"],
    "explorer.exe": ["open file explorer", "browse files", "open windows explorer", "show my files", "open computer"],
    "mspaint.exe": ["open paint", "I need to draw something", "launch ms paint", "create a drawing", "open an image editor"],
    "taskmgr.exe": ["open task manager", "check performance", "kill a process", "show running apps", "taskmgr please"],
    "control.exe": ["open control panel", "change system settings", "system config", "launch control panel"],
    "chrome.exe": ["open chrome", "launch google chrome", "browse with chrome", "open google chrome"],
    "spotify.exe": ["play some music", "open spotify", "launch spotify", "I want to listen to songs"],
    "code.exe": ["open vs code", "launch visual studio code", "I need to code", "open code editor"]
}

variations = [
    "{}",
    "Can you {}",
    "Please {}",
    "I need to {}",
    "{} right now",
    "Help me {}",
    "Quickly {}"
]

dataset = []
for app, intents in apps.items():
    for intent in intents:
        for var in variations:
            instruction = var.format(intent)
            dataset.append({
                "instruction": instruction,
                "output": f"start {app}"
            })

# Shuffle to make it diverse
random.shuffle(dataset)

with open("launcher_dataset.jsonl", "w") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")

print(f"Generated {len(dataset)} examples in launcher_dataset.jsonl")
