"""
Vision-based UI Parser for locating elements on screen.
Uses OpenCV and Tesseract OCR, with a fallback to Windows UIAutomation.
"""
# pylint: disable=no-member, too-many-locals
import os
from typing import Optional, Tuple

import cv2
import mss
import numpy as np
import pytesseract
import uiautomation as auto

from core.logger import setup_logger

logger = setup_logger("UIParser")


class UIParser: # pylint: disable=too-few-public-methods
    """
    Parses the screen to find the coordinates of UI elements based on text descriptions.
    """
    def __init__(self) -> None:
        auto.SetGlobalSearchTimeout(2.0)
        self.sct = mss.mss()

        # Explicitly set the Tesseract path so it works immediately without a
        # terminal restart
        tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def find_element(
            self, element_description: str) -> Optional[Tuple[int, int]]:
        """
        Uses OCR as the primary method to find text on screen.
        Falls back to Windows Accessibility Tree if OCR fails.
        """
        logger.info(
            "Scanning for '%s' using Vision OCR...", element_description)
        coords = self._find_element_ocr(element_description)

        if coords:
            return coords

        logger.warning(
            "OCR failed to find '%s'. Falling back to Accessibility Tree...", element_description)
        return self._find_element_accessibility(element_description)

    def _find_element_ocr(
            self, element_description: str) -> Optional[Tuple[int, int]]: # pylint: disable=too-many-locals
        """
        Captures the screen and uses OpenCV + Tesseract to find bounding boxes of text.
        """
        try:
            # 1. Capture screen
            monitor = self.sct.monitors[1]  # primary monitor
            screenshot = self.sct.grab(monitor)

            # Convert to numpy array for OpenCV
            img = np.array(screenshot)

            # Convert from BGRA (mss default) to BGR, then to Grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

            # 2. Pre-processing: Apply thresholding to make text pop
            # We use an inverted binary threshold combined with Otsu's method
            # to handle varying backgrounds (dark mode vs light mode).
            _, thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # 3. Run Tesseract to get data
            # psm 11 means: Sparse text. Find as much text as possible in no
            # particular order.
            custom_config = r'--oem 3 --psm 11'
            data = pytesseract.image_to_data(
                thresh, output_type=pytesseract.Output.DICT, config=custom_config)

            target_text = element_description.lower().strip()
            best_match_idx = -1

            # 4. Iterate through all OCR results
            for i, text_val in enumerate(data['text']):
                conf = int(data['conf'][i])
                text = text_val.lower().strip()

                # Filter out low confidence or empty text
                if conf > 30 and text:
                    # Fuzzy match: Exact or substring match
                    if target_text == text:
                        best_match_idx = i
                        break
                    # Only allow substring matching if the matched part is somewhat significant (e.g., >3 chars)
                    # to prevent tiny artifacts like "in" from matching "result_link"
                    elif len(text) >= 4 and (target_text in text or text in target_text):
                        best_match_idx = i
                        break
                    else:
                        # Fallback: Word overlap matching. Useful if LLM asks for "Google Search results: Gemini AI"
                        # but the screen just says "Gemini - Your AI assistant" or "Gemini AI".
                        target_words = set([w for w in target_text.split() if len(w) > 3])
                        screen_words = set([w for w in text.split() if len(w) > 3])
                        if target_words and screen_words:
                            overlap = len(target_words.intersection(screen_words))
                            if overlap >= 2 or (len(target_words) == 1 and overlap == 1):
                                best_match_idx = i
                                break

            if best_match_idx != -1:
                x = data['left'][best_match_idx]
                y = data['top'][best_match_idx]
                w = data['width'][best_match_idx]
                h = data['height'][best_match_idx]

                # Calculate center coordinates for clicking
                center_x = x + (w // 2)
                center_y = y + (h // 2)

                # Adjust for monitor offsets if necessary
                center_x += monitor['left']
                center_y += monitor['top']

                logger.info(
                    "OCR Found '%s' matching '%s' at (%s, %s)",
                    data['text'][best_match_idx], element_description, center_x, center_y)
                return (center_x, center_y)

            return None

        except Exception as e: # pylint: disable=broad-exception-caught
            logger.error("OCR Parsing failed: %s", e)
            logger.debug("Did you install Tesseract-OCR on your system?")
            return None

    def _find_element_accessibility(
            self, element_description: str) -> Optional[Tuple[int, int]]:
        """Fallback method using the Windows UIAutomation tree."""
        desc_lower = element_description.lower()
        desktop = auto.GetRootControl()
        best_match = None

        try:
            # 1st attempt: exact match by name
            control = auto.Control(Name=element_description)
            if control.Exists(1, 1):
                best_match = control

            # 2nd attempt: partial match walk tree
            if not best_match:
                for c, _ in auto.WalkTree(desktop, maxDepth=3):
                    if c.Name and desc_lower in c.Name.lower():
                        best_match = c
                        break

            if best_match:
                rect = best_match.BoundingRectangle
                x = (rect.left + rect.right) // 2
                y = (rect.top + rect.bottom) // 2
                logger.info(
                    "Accessibility Tree Found '%s' at (%s, %s)", element_description, x, y)
                return (x, y)

            logger.warning(
                "Could not find '%s' using any method.", element_description)
            return None

        except Exception as e: # pylint: disable=broad-exception-caught
            logger.error("Accessibility Parsing failed: %s", e)
            return None
