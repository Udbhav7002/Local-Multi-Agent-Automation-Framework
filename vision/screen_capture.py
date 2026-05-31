"""
Screen capture utility using the `mss` library.
Provides extremely fast, native screenshot capabilities.
"""
import mss
import mss.tools

from core.logger import setup_logger

logger = setup_logger("ScreenCapture")


class ScreenCapture: # pylint: disable=too-few-public-methods
    """
    Handles taking fullscreen screenshots of the primary monitor.
    """
    def __init__(self) -> None:
        self.sct = mss.mss()

    def capture_fullscreen(self, output_filename: str = "screenshot.png") -> str:
        """
        Captures the entire screen and saves it to a PNG file.
        Returns the path to the saved image.
        """
        try:
            monitor = self.sct.monitors[1]
            output = self.sct.grab(monitor)
            mss.tools.to_png(output.rgb, output.size, output=output_filename)
            logger.debug("Captured screenshot to %s", output_filename)
            return output_filename
        except Exception as e: # pylint: disable=broad-exception-caught
            logger.error("Failed to capture screen: %s", e)
            raise e
