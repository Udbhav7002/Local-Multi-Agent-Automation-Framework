import uiautomation as auto
import time

class UIParser:
    def __init__(self):
        # We configure uiautomation settings if needed
        auto.SetGlobalSearchTimeout(2.0)

    def find_element(self, element_description: str):
        """
        Uses the Windows Accessibility Tree to find an element matching the description.
        Returns (x, y) coordinates of the center of the element, or None if not found.
        """
        print(f"[VISION] Scanning Accessibility Tree for '{element_description}'...")
        
        # Heuristics: search by Name or ClassName
        # Try to find a control whose name contains the description (case insensitive)
        desc_lower = element_description.lower()
        
        # We start searching from the desktop root
        desktop = auto.GetRootControl()
        
        best_match = None
        try:
            # We iterate through some common controls on screen.
            # Using WalkTree is very thorough but can be slow. 
            # We use a breadth-first search approach or rely on auto's search.
            
            # 1st attempt: exact match by name
            control = auto.Control(Name=element_description)
            if control.Exists(1, 1):
                best_match = control
            
            if not best_match:
                for c, depth in auto.WalkTree(desktop, maxDepth=3):
                    if c.Name and desc_lower in c.Name.lower():
                        best_match = c
                        break
                        
            if best_match:
                rect = best_match.BoundingRectangle
                x = (rect.left + rect.right) // 2
                y = (rect.top + rect.bottom) // 2
                print(f"[VISION] Found '{element_description}' at ({x}, {y})")
                return (x, y)
                
            print(f"[VISION] Could not find '{element_description}' on screen.")
            return None
            
        except Exception as e:
            print(f"[ERROR] UI Parsing failed: {str(e)}")
            return None
