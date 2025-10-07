"""UI components for object selection and reference point selection."""

import cv2
import numpy as np
from typing import Optional, Tuple, Callable


class ReferencePointSelector:
    """UI for selecting which point on the bounding box to track."""

    REFERENCE_POINTS = {
        '1': ('top-left', lambda x, y, w, h: (x, y)),
        '2': ('top-center', lambda x, y, w, h: (x + w // 2, y)),
        '3': ('top-right', lambda x, y, w, h: (x + w, y)),
        '4': ('center-left', lambda x, y, w, h: (x, y + h // 2)),
        '5': ('center', lambda x, y, w, h: (x + w // 2, y + h // 2)),
        '6': ('center-right', lambda x, y, w, h: (x + w, y + h // 2)),
        '7': ('bottom-left', lambda x, y, w, h: (x, y + h)),
        '8': ('bottom-center', lambda x, y, w, h: (x + w // 2, y + h)),
        '9': ('bottom-right', lambda x, y, w, h: (x + w, y + h)),
    }

    def __init__(self):
        self.selected_point = 'bottom-center'  # Default
        self.selected_key = '8'

    def select_reference_point(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[str, Callable]:
        """Display UI for selecting reference point on bounding box.

        Args:
            frame: Video frame
            bbox: Bounding box (x, y, w, h)

        Returns:
            Tuple of (point_name, point_calculator_function)
        """
        x, y, w, h = bbox
        self.frame = frame.copy()

        cv2.namedWindow("Select Reference Point")

        print("\n=== Select Tracking Reference Point ===")
        print("Choose which point of the bounding box to track:")
        print("  1: Top-Left      2: Top-Center      3: Top-Right")
        print("  4: Center-Left   5: Center          6: Center-Right")
        print("  7: Bottom-Left   8: Bottom-Center   9: Bottom-Right")
        print("\nPress the corresponding number key (default: 8 = Bottom-Center)")
        print("Press ENTER to confirm")

        while True:
            display_frame = self.frame.copy()

            # Draw bounding box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw all reference points
            for key, (name, calc_func) in self.REFERENCE_POINTS.items():
                pt_x, pt_y = calc_func(x, y, w, h)
                color = (0, 0, 255) if key == self.selected_key else (128, 128, 128)
                size = 8 if key == self.selected_key else 4
                cv2.circle(display_frame, (pt_x, pt_y), size, color, -1)
                cv2.putText(display_frame, key, (pt_x + 10, pt_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Show selected point
            cv2.putText(display_frame, f"Selected: {self.selected_point} ({self.selected_key})",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press ENTER to confirm",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Select Reference Point", display_frame)
            key = cv2.waitKey(1) & 0xFF

            # Number key pressed
            if chr(key) in self.REFERENCE_POINTS:
                self.selected_key = chr(key)
                self.selected_point = self.REFERENCE_POINTS[self.selected_key][0]

            # Enter key - confirm
            elif key == 13:
                break

        cv2.destroyAllWindows()
        print(f"✓ Selected reference point: {self.selected_point}")

        return self.selected_point, self.REFERENCE_POINTS[self.selected_key][1]


class ObjectSelector:
    """UI for selecting an object to track."""

    def __init__(self):
        self.bbox = None
        self.start_point = None
        self.end_point = None
        self.selecting = False
        self.frame = None
        self.clone = None

    def select_roi(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Display UI for user to select region of interest."""
        self.frame = frame.copy()
        self.clone = frame.copy()
        self.bbox = None
        self.start_point = None
        self.end_point = None
        self.selecting = False

        cv2.namedWindow("Select Object to Track")
        cv2.setMouseCallback("Select Object to Track", self._mouse_callback)

        print("\n=== Object Selection ===")
        print("1. Click and drag to draw a box around the object")
        print("2. Press ENTER to confirm selection")
        print("3. Press 'r' to reset selection")
        print("4. Press ESC to cancel")

        while True:
            cv2.imshow("Select Object to Track", self.frame)
            key = cv2.waitKey(1) & 0xFF

            # Enter key - confirm selection
            if key == 13 and self.bbox is not None:
                break
            # ESC key - cancel
            elif key == 27:
                self.bbox = None
                break
            # 'r' key - reset
            elif key == ord('r'):
                self.frame = self.clone.copy()
                self.bbox = None
                self.start_point = None
                self.end_point = None
                self.selecting = False

        cv2.destroyAllWindows()
        return self.bbox

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.start_point = (x, y)
            self.end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selecting:
                self.end_point = (x, y)
                # Draw rectangle on temporary frame
                self.frame = self.clone.copy()
                cv2.rectangle(self.frame, self.start_point, self.end_point, (0, 255, 0), 2)
                cv2.putText(self.frame, "Release mouse to set selection",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False
            self.end_point = (x, y)

            # Calculate bounding box (x, y, w, h)
            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])

            w = x2 - x1
            h = y2 - y1

            if w > 5 and h > 5:  # Minimum size check
                self.bbox = (x1, y1, w, h)
                # Draw final rectangle
                self.frame = self.clone.copy()
                cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(self.frame, "Press ENTER to confirm, 'r' to reset",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
