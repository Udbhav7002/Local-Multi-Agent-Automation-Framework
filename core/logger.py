"""
Logger module for standardized output across all framework components.
Uses rich for a clean, premium terminal look without emojis.
"""
import logging
import os

from rich.logging import RichHandler
from rich.console import Console

# Share a single global console for the entire app
console = Console()

def setup_logger(name: str) -> logging.Logger:
    """Creates a configured logger with standard formatting."""
    logger = logging.getLogger(name)

    # Prevent adding handlers multiple times if instantiated multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(
        logging.DEBUG if os.getenv(
            "DEBUG",
            "False").lower() in (
            "true",
            "1",
            "t") else logging.INFO)

    # Rich handler for clean, colorized terminal output
    handler = RichHandler(
        console=console,
        show_path=False,
        show_time=True,
        rich_tracebacks=True,
        markup=True
    )
    
    # We let rich handle the formatting mostly, but we can set a basic format string
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)

    return logger
