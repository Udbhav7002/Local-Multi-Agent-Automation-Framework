"""
Configuration module for the Local Multi-Agent Automation Framework.
"""
import os
from dataclasses import dataclass


@dataclass
class Config:
    """Central configuration for the framework."""
    # Models
    manager_model: str = os.getenv("MANAGER_MODEL", "llama3:latest")    # Planner model
    worker_model: str = os.getenv("WORKER_MODEL", "llama3:latest")      # Fast local model
    vision_model: str = os.getenv("VISION_MODEL", "llava:latest")

    # Retry logic
    max_plan_regenerations: int = int(os.getenv("MAX_PLAN_REGENERATIONS", "3"))
    max_step_retries: int = int(os.getenv("MAX_STEP_RETRIES", "1"))

    # Behavior
    auto_continue: bool = os.getenv(
        "AUTO_CONTINUE", "False").lower() in (
        "true", "1", "yes", "y")

    # API endpoints
    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL", "http://localhost:11434")

config = Config()
