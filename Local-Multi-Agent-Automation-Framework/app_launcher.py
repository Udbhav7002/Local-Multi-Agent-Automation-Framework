import subprocess
import requests
import json
import sys

OLLAMA_URL = "http://localhost:11434/api/generate"
# Assuming the user creates an Ollama modelfile named 'fast-launcher'
# For example: ollama create fast-launcher -f Modelfile
MODEL_NAME = "fast-launcher" 

SYSTEM_PROMPT = "You are an ultra-fast Windows application launcher. Your ONLY job is to output the exact Windows command to fulfill the user's request. Do not output any conversational text, explanations, or formatting. Just the raw command."

def query_launcher(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 50
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        print("Make sure Ollama is running and the 'fast-launcher' model is loaded.")
        sys.exit(1)

def main():
    print("🚀 Ultra-Fast Windows App Launcher (Type 'exit' to quit)")
    print(f"Model: {MODEL_NAME}")
    while True:
        try:
            user_input = input("\nLauncher> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                break
                
            print("Thinking...", end="\r")
            command = query_launcher(user_input)
            
            # Since the model is fine-tuned to only output the command, we print it directly
            print(f"Executing: {command}                                      ")
            
            # Execute the returned command natively
            # SECURITY WARNING: subprocess.run with shell=True is dangerous. 
            # We trust the fine-tuned model to only output safe 'start' commands.
            subprocess.run(command, shell=True)
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
