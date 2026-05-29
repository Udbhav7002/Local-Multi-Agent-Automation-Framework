# vision/screen_capture.py
import mss
import mss.tools

class ScreenCapture:
    def __init__(self):
        self.sct = mss.mss()

    def capture_fullscreen(self, output_filename="screenshot.png"):
        monitor = self.sct.monitors[1]
        output = self.sct.grab(monitor)
        mss.tools.to_png(output.rgb, output.size, output=output_filename)
        return output_filename
