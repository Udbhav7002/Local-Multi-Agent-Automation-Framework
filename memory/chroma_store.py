"""
Memory module for storing and retrieving successful and failed plans using ChromaDB.
"""
import uuid

import chromadb

from core.logger import setup_logger

logger = setup_logger("ChromaStore")


class ChromaStore:
    """
    Handles interactions with ChromaDB for semantic memory retrieval.
    Stores past execution plans and their success/failure states.
    """
    def __init__(self) -> None:
        # Initialize chromadb persistent client
        self.client = chromadb.PersistentClient(path="./chroma_db")
        # Collections
        self.success_collection = self.client.get_or_create_collection(
            name="successful_plans")
        self.failure_collection = self.client.get_or_create_collection(
            name="failed_plans")

    def get_plan(self, task_description: str) -> str | None:
        """Query ChromaDB for an exact or highly similar task."""
        try:
            results = self.success_collection.query(
                query_texts=[task_description],
                n_results=1
            )
            if results and results.get('documents') and len(
                    results['documents'][0]) > 0:
                # Check distance to ensure high confidence
                distance = results['distances'][0][0]
                if distance < 0.2:  # Very similar
                    logger.debug(
                        "Found cached plan with distance: %s", distance)
                    return results['documents'][0][0]
            return None
        except Exception as e: # pylint: disable=broad-exception-caught
            logger.error("Error querying success collection: %s", e)
            return None

    def save_plan(self, task_description: str, plan_json: str) -> None:
        """Save a successful plan to ChromaDB using UUIDs."""
        try:
            doc_id = str(uuid.uuid4())
            self.success_collection.add(
                documents=[plan_json],
                metadatas=[{"task": task_description}],
                ids=[doc_id]
            )
            logger.debug(
                "Successfully saved plan to ChromaStore with ID: %s", doc_id)
        except Exception as e: # pylint: disable=broad-exception-caught
            logger.error("Error saving to success collection: %s", e)

    def get_failures(self, task_description: str) -> list:
        """Get past failed strategies for a similar task."""
        try:
            results = self.failure_collection.query(
                query_texts=[task_description],
                n_results=3
            )
            failures = []
            if results and results.get('documents') and len(
                    results['documents']) > 0:
                for idx, doc in enumerate(results['documents'][0]):
                    distance = results['distances'][0][idx]
                    if distance < 0.3:  # Slightly looser threshold for failures
                        reason = results['metadatas'][0][idx].get(
                            'reason', 'Unknown reason')
                        failures.append({'plan': doc, 'reason': reason})
            return failures
        except Exception as e: # pylint: disable=broad-exception-caught
            logger.error("Error querying failure collection: %s", e)
            return []

    def save_failure(
            self,
            task_description: str,
            failed_plan: str,
            reason: str) -> None:
        """Save a failed plan to ChromaDB."""
        try:
            doc_id = str(uuid.uuid4())
            self.failure_collection.add(
                documents=[failed_plan],
                metadatas=[{"task": task_description, "reason": reason}],
                ids=[doc_id]
            )
            logger.info(
                "Logged failed plan to Reflection Memory to prevent future recurrence.")
        except Exception as e: # pylint: disable=broad-exception-caught
            logger.error("Error saving to failure collection: %s", e)
