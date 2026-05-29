import chromadb
import json

class ChromaStore:
    def __init__(self):
        # Initialize chromadb persistent client
        self.client = chromadb.PersistentClient(path="./chroma_db")
        # Create or get collection for plans
        self.collection = self.client.get_or_create_collection(name="successful_plans")

    def get_plan(self, task_description: str):
        """Query ChromaDB for an exact or highly similar task."""
        results = self.collection.query(
            query_texts=[task_description],
            n_results=1
        )
        if results and results['documents'] and len(results['documents'][0]) > 0:
            # Check distance to ensure high confidence
            distance = results['distances'][0][0]
            if distance < 0.2: # Very similar
                return results['documents'][0][0]
        return None

    def save_plan(self, task_description: str, plan_json: str):
        """Save a successful plan to ChromaDB."""
        # Use the task description as the ID (hashed or raw, raw is fine for local)
        import hashlib
        doc_id = hashlib.md5(task_description.encode()).hexdigest()
        
        self.collection.add(
            documents=[plan_json],
            metadatas=[{"task": task_description}],
            ids=[doc_id]
        )
