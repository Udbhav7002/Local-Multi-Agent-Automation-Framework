import ollama
import asyncio

class Router:
    def __init__(self, model_name="llama3.2:1b"):
        self.model_name = model_name

    async def classify(self, user_text: str):
        """
        Classifies user text as either 'question' or 'task'.
        Returns a tuple: (intent, answer_if_question)
        """
        # Deterministic Heuristic Bypass
        action_verbs = ['open', 'run', 'start', 'click', 'type', 'search', 'delete', 'create', 'make', 'do', 'close', 'kill']
        first_word = user_text.strip().split()[0].lower() if user_text.strip() else ""
        if first_word in action_verbs:
            return "task", None

        system_prompt = """
        You are a routing agent for a local OS controller. 
        Classify the user's input as either 'question' or 'task'.
        
        - 'task': The user is asking you to perform a physical action on the computer, like opening an app, running a command, clicking, or managing files.
        - 'question': The user is just chatting, greeting, or asking for information.
        
        EXAMPLES:
        User: hi
        Output: question
        User: open brave
        Output: task
        User: what is the weather?
        Output: question
        User: delete the temp folder
        Output: task
        
        Output ONLY the word 'question' or 'task' and nothing else.
        """
        
        try:
            # We use a blocking ollama call in a thread to keep async event loop free
            response = await asyncio.to_thread(
                ollama.chat, 
                model=self.model_name, 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ]
            )
            
            intent = response['message']['content'].strip().lower()
            
            # If the model explicitly says task, we treat it as a task.
            if "task" in intent and "question" not in intent:
                return "task", None
            elif intent.startswith("task"):
                return "task", None
                
            # Otherwise, default to question/chat behavior
            ans_response = await asyncio.to_thread(
                ollama.chat,
                model=self.model_name, # In prod, use a larger model like llama3.1:8b for answering
                messages=[{"role": "user", "content": user_text}]
            )
            return "question", ans_response['message']['content']
            
        except Exception as e:
            # Default to task if ollama fails (or log error)
            return "task", str(e)
