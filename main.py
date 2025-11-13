import json
import re

# from norfair import Detection, Tracker, Video
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class MeterStickCalibrator:
    """UI for calibrating scale using a meter stick with adjustable tick marks."""

    def __init__(self):
        self.start_point = None
        self.end_point = None
        self.selecting = False
        self.frame = None
        self.clone = None
        self.tick_positions = []  # List of [x, y, cm_value]
        self.selected_tick = None
        self.dragging = False
        self.stick_length_cm = 100.0
        self.needs_redraw = False

    def calibrate(
        self, frame: np.ndarray, stick_length_cm: float = 100.0
    ) -> Optional[dict]:
        """Display UI for user to mark the meter stick endpoints and adjust ticks.

        Args:
            frame: Video frame containing the meter stick
            stick_length_cm: Length of the reference stick in centimeters (default 100 for meter stick)

        Returns:
            Dictionary with tick positions and calibration data, or None if cancelled
        """
        self.frame = frame.copy()
        self.clone = frame.copy()
        self.start_point = None
        self.end_point = None
        self.selecting = False
        self.stick_length_cm = stick_length_cm
        self.tick_positions = []

        cv2.namedWindow("Calibrate Scale - Mark Reference Stick")
        cv2.setMouseCallback("Calibrate Scale - Mark Reference Stick", self._mouse_callback)

        print("\n=== Scale Calibration - Step 1 ===")
        print(f"Reference length: {stick_length_cm} cm")
        print("1. Click at one end of the reference stick")
        print("2. Click at the other end of the reference stick")
        print("3. Press ENTER to continue")
        print("4. Press 'r' to reset")
        print("5. Press ESC to cancel")

        while True:
            cv2.imshow("Calibrate Scale - Mark Reference Stick", self.frame)
            key = cv2.waitKey(1) & 0xFF

            # Enter key - confirm selection
            if (
                key == 13
                and self.start_point is not None
                and self.end_point is not None
            ):
                break
            # ESC key - cancel
            elif key == 27:
                cv2.destroyAllWindows()
                return None
            # 'r' key - reset
            elif key == ord("r"):
                self.frame = self.clone.copy()
                self.start_point = None
                self.end_point = None
                self.selecting = False

        # Initialize tick positions
        self._initialize_tick_positions()

        # Now allow tick adjustment
        if not self._adjust_ticks():
            cv2.destroyAllWindows()
            return None

        cv2.destroyAllWindows()

        # Build calibration data
        calibration_data = {
            "tick_positions": self.tick_positions,
            "start_point": self.start_point,
            "end_point": self.end_point,
            "stick_length_cm": stick_length_cm,
        }

        print(f"\n✓ Calibration complete!")
        print(f"  Total tick marks: {len(self.tick_positions)}")

        return calibration_data

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for meter stick marking."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.start_point is None:
                # First click - set start point
                self.start_point = (x, y)
                self.frame = self.clone.copy()
                cv2.circle(self.frame, self.start_point, 5, (0, 255, 0), -1)
                cv2.putText(
                    self.frame,
                    "Click at the other end",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            else:
                # Second click - set end point
                self.end_point = (x, y)
                self._draw_measurement()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.start_point is not None and self.end_point is None:
                # Show preview line while moving
                temp_frame = self.clone.copy()
                cv2.circle(temp_frame, self.start_point, 5, (0, 255, 0), -1)
                cv2.line(temp_frame, self.start_point, (x, y), (255, 255, 0), 2)

                # Calculate and display distance
                dx = x - self.start_point[0]
                dy = y - self.start_point[1]
                distance = np.sqrt(dx**2 + dy**2)
                cv2.putText(
                    temp_frame,
                    f"{distance:.1f} pixels",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

                self.frame = temp_frame

    def _draw_measurement(self):
        """Draw the final measurement line with tick marks every 10cm."""
        self.frame = self.clone.copy()
        cv2.circle(self.frame, self.start_point, 5, (0, 255, 0), -1)
        cv2.circle(self.frame, self.end_point, 5, (0, 255, 0), -1)
        cv2.line(self.frame, self.start_point, self.end_point, (0, 255, 0), 2)

        # Calculate distance
        dx = self.end_point[0] - self.start_point[0]
        dy = self.end_point[1] - self.start_point[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Draw tick marks every 10cm up to stick length
        num_ticks = int(self.stick_length_cm / 10) + 1  # e.g., 11 ticks for 100cm (0, 10, ..., 100)
        for i in range(num_ticks):
            t = i / (num_ticks - 1)  # Parameter from 0 to 1

            # Position along the line
            tick_x = int(self.start_point[0] + t * dx)
            tick_y = int(self.start_point[1] + t * dy)

            # Perpendicular direction for tick mark
            length = 10 if i % 2 == 0 else 5  # Longer tick every 20cm
            perp_dx = -dy / distance * length
            perp_dy = dx / distance * length

            # Draw tick mark
            tick_start = (int(tick_x - perp_dx), int(tick_y - perp_dy))
            tick_end = (int(tick_x + perp_dx), int(tick_y + perp_dy))
            cv2.line(self.frame, tick_start, tick_end, (0, 255, 255), 2)

            # Draw label for major ticks
            if i % 2 == 0:
                label = f"{i * 10}"
                cv2.putText(
                    self.frame,
                    label,
                    (tick_x + 5, tick_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                )

        # Draw distance text
        mid_x = (self.start_point[0] + self.end_point[0]) // 2
        mid_y = (self.start_point[1] + self.end_point[1]) // 2
        cv2.putText(
            self.frame,
            f"{distance:.1f} pixels",
            (mid_x + 10, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            self.frame,
            "Press ENTER to adjust ticks, 'r' to reset",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    def _initialize_tick_positions(self):
        """Initialize tick positions along the line."""
        dx = self.end_point[0] - self.start_point[0]
        dy = self.end_point[1] - self.start_point[1]

        num_ticks = int(self.stick_length_cm / 10) + 1  # e.g., 11 ticks for 100cm (0, 10, ..., 100)
        self.tick_positions = []
        for i in range(num_ticks):
            t = i / (num_ticks - 1)
            tick_x = int(self.start_point[0] + t * dx)
            tick_y = int(self.start_point[1] + t * dy)
            cm_value = i * 10
            self.tick_positions.append([tick_x, tick_y, cm_value])

    def _adjust_ticks(self) -> bool:
        """Allow user to adjust tick mark positions for lens distortion correction."""
        print("\n=== Scale Calibration - Step 2: Adjust Tick Marks ===")
        print("The tick marks are now displayed. Adjust them if needed:")
        print("  - Click and drag any tick mark to adjust its position")
        print("  - Press ENTER when satisfied")
        print("  - Press ESC to cancel")

        cv2.setMouseCallback(
            "Calibrate Scale - Mark Reference Stick", self._tick_adjust_callback
        )

        # Initial draw
        self._draw_adjustable_ticks()
        cv2.imshow("Calibrate Scale - Mark Reference Stick", self.frame)
        self.needs_redraw = False

        while True:
            # Only redraw if something changed
            if self.needs_redraw:
                self._draw_adjustable_ticks()
                cv2.imshow("Calibrate Scale - Mark Reference Stick", self.frame)
                self.needs_redraw = False

            key = cv2.waitKey(10) & 0xFF

            if key == 13:  # Enter - confirm
                return True
            elif key == 27:  # ESC - cancel
                return False

    def _draw_adjustable_ticks(self):
        """Draw tick marks with adjustment handles."""
        self.frame = self.clone.copy()
        cv2.circle(self.frame, self.start_point, 5, (0, 255, 0), -1)
        cv2.circle(self.frame, self.end_point, 5, (0, 255, 0), -1)

        # Draw line connecting tick marks
        for i in range(len(self.tick_positions) - 1):
            pt1 = (self.tick_positions[i][0], self.tick_positions[i][1])
            pt2 = (self.tick_positions[i + 1][0], self.tick_positions[i + 1][1])
            cv2.line(self.frame, pt1, pt2, (0, 255, 0), 1)

        # Draw each tick mark
        for i, (tick_x, tick_y, cm_value) in enumerate(self.tick_positions):
            # Calculate perpendicular direction
            if i < len(self.tick_positions) - 1:
                next_x, next_y, _ = self.tick_positions[i + 1]
                dx_local = next_x - tick_x
                dy_local = next_y - tick_y
            elif i > 0:
                prev_x, prev_y, _ = self.tick_positions[i - 1]
                dx_local = tick_x - prev_x
                dy_local = tick_y - prev_y
            else:
                dx_local = self.end_point[0] - self.start_point[0]
                dy_local = self.end_point[1] - self.start_point[1]

            local_dist = np.sqrt(dx_local**2 + dy_local**2)
            if local_dist > 0:
                length = 10 if i % 2 == 0 else 5
                perp_dx = -dy_local / local_dist * length
                perp_dy = dx_local / local_dist * length

                # Draw tick mark
                tick_start = (int(tick_x - perp_dx), int(tick_y - perp_dy))
                tick_end = (int(tick_x + perp_dx), int(tick_y + perp_dy))

                # Highlight selected tick
                color = (255, 0, 255) if self.selected_tick == i else (0, 255, 255)
                cv2.line(self.frame, tick_start, tick_end, color, 2)

                # Draw draggable handle
                handle_color = (
                    (255, 0, 255) if self.selected_tick == i else (0, 255, 255)
                )
                cv2.circle(self.frame, (tick_x, tick_y), 6, handle_color, -1)

                # Draw label for major ticks
                if i % 2 == 0:
                    label = f"{int(cm_value)}"
                    cv2.putText(
                        self.frame,
                        label,
                        (tick_x + 5, tick_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 255),
                        1,
                    )

        cv2.putText(
            self.frame,
            "Drag tick marks to adjust for lens distortion. Press ENTER when done.",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    def _tick_adjust_callback(self, event, x, y, flags, param):
        """Handle mouse events for tick adjustment."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Find closest tick mark
            min_dist = float("inf")
            closest_tick = None
            for i, (tick_x, tick_y, _) in enumerate(self.tick_positions):
                dist = np.sqrt((x - tick_x) ** 2 + (y - tick_y) ** 2)
                if dist < min_dist and dist < 15:  # Within 15 pixels
                    min_dist = dist
                    closest_tick = i

            if closest_tick is not None:
                self.selected_tick = closest_tick
                self.dragging = True
                self.needs_redraw = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and self.selected_tick is not None:
                # Update tick position
                self.tick_positions[self.selected_tick][0] = x
                self.tick_positions[self.selected_tick][1] = y
                self.needs_redraw = True

        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging:
                self.dragging = False
                self.selected_tick = None
                self.needs_redraw = True


class TimerCalibrator:
    """UI for selecting the region of the on-screen timer for OCR."""

    def __init__(self):
        self.bbox = None
        self.start_point = None
        self.end_point = None
        self.selecting = False
        self.frame = None
        self.clone = None
        self.rotation = 0  # Rotation angle: 0, 90, 180, or 270

    def calibrate(self, frame: np.ndarray) -> Optional[Tuple[Tuple[int, int, int, int], int]]:
        """Display UI for user to select the timer display region.

        Args:
            frame: Video frame containing the timer display

        Returns:
            Tuple of (bbox, rotation) where bbox is (x, y, w, h) and rotation is 0/90/180/270,
            or None if cancelled
        """
        if not TESSERACT_AVAILABLE:
            print("\n⚠️  pytesseract not available. Timer feature disabled.")
            print("Install with: pip install pytesseract")
            print("Also ensure Tesseract OCR is installed on your system.")
            return None

        self.frame = frame.copy()
        self.clone = frame.copy()
        self.bbox = None
        self.start_point = None
        self.end_point = None
        self.selecting = False

        cv2.namedWindow("Select Timer Region")
        cv2.setMouseCallback("Select Timer Region", self._mouse_callback)

        print("\n=== Timer Region Selection ===")
        print("1. Click and drag to draw a box around the timer display")
        print("2. Press '0', '9', '1', or '2' to rotate: 0°, 90°, 180°, 270°")
        print("3. Press ENTER to confirm selection")
        print("4. Press 'r' to reset selection")
        print("5. Press ESC to skip timer tracking")

        while True:
            # Update display with rotation info
            display_frame = self.frame.copy()
            if self.bbox is not None:
                cv2.putText(
                    display_frame,
                    f"Rotation: {self.rotation}° (press 0/9/1/2 to change)",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("Select Timer Region", display_frame)
            key = cv2.waitKey(1) & 0xFF

            # Enter key - confirm selection
            if key == 13 and self.bbox is not None:
                # Test OCR on selected region with rotation
                x, y, w, h = self.bbox
                timer_roi = self.clone[y:y+h, x:x+w]
                test_text = self._ocr_timer(timer_roi, self.rotation)

                print(f"\n✓ Timer region selected")
                print(f"  Rotation: {self.rotation}°")
                print(f"  Test OCR result: '{test_text}'")
                print("  (Make sure the timer digits are readable)")
                break
            # ESC key - cancel
            elif key == 27:
                self.bbox = None
                break
            # 'r' key - reset
            elif key == ord("r"):
                self.frame = self.clone.copy()
                self.bbox = None
                self.start_point = None
                self.end_point = None
                self.selecting = False
                self.rotation = 0
            # Rotation keys: 0=0°, 9=90°, 1=180°, 2=270°
            elif key == ord("0"):
                self.rotation = 0
            elif key == ord("9"):
                self.rotation = 90
            elif key == ord("1"):
                self.rotation = 180
            elif key == ord("2"):
                self.rotation = 270

        cv2.destroyAllWindows()
        return (self.bbox, self.rotation) if self.bbox is not None else None

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for timer region selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.start_point = (x, y)
            self.end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selecting:
                self.end_point = (x, y)
                # Draw rectangle on temporary frame
                self.frame = self.clone.copy()
                cv2.rectangle(
                    self.frame, self.start_point, self.end_point, (0, 255, 255), 2
                )
                cv2.putText(
                    self.frame,
                    "Release mouse to set timer region",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

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
                cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    self.frame,
                    "Press ENTER to confirm, 'r' to reset",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

    @staticmethod
    def _ocr_timer(timer_roi: np.ndarray, rotation: int = 0) -> Optional[float]:
        """Extract timestamp from timer region using OCR.

        Args:
            timer_roi: Cropped image of timer display
            rotation: Rotation angle (0, 90, 180, or 270 degrees)

        Returns:
            Timestamp in seconds, or None if parsing failed
        """
        if not TESSERACT_AVAILABLE:
            return None

        # Apply rotation if needed
        if rotation == 90:
            timer_roi = cv2.rotate(timer_roi, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            timer_roi = cv2.rotate(timer_roi, cv2.ROTATE_180)
        elif rotation == 270:
            timer_roi = cv2.rotate(timer_roi, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Preprocess image for better OCR
        gray = cv2.cvtColor(timer_roi, cv2.COLOR_BGR2GRAY)
        # Increase contrast
        gray = cv2.equalizeHist(gray)
        # Apply threshold to get black text on white background
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Configure tesseract for digits and common time separators
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:.'

        try:
            text = pytesseract.image_to_string(thresh, config=custom_config).strip()

            # Parse common time formats: MM:SS.mmm, SS.mmm, M:SS, etc.
            # Try to extract numbers and convert to seconds

            # Format: MM:SS.mmm or MM:SS
            match = re.match(r'(\d+):(\d+)\.?(\d*)', text)
            if match:
                minutes = int(match.group(1))
                seconds = int(match.group(2))
                milliseconds = int(match.group(3)) if match.group(3) else 0
                # Normalize milliseconds based on number of digits
                if len(match.group(3)) == 3:
                    milliseconds = milliseconds
                elif len(match.group(3)) == 2:
                    milliseconds = milliseconds * 10
                elif len(match.group(3)) == 1:
                    milliseconds = milliseconds * 100
                return minutes * 60 + seconds + milliseconds / 1000.0

            # Format: SS.mmm or SS
            match = re.match(r'(\d+)\.?(\d*)', text)
            if match:
                seconds = int(match.group(1))
                milliseconds = int(match.group(2)) if match.group(2) else 0
                # Normalize milliseconds
                if len(match.group(2)) == 3:
                    milliseconds = milliseconds
                elif len(match.group(2)) == 2:
                    milliseconds = milliseconds * 10
                elif len(match.group(2)) == 1:
                    milliseconds = milliseconds * 100
                return seconds + milliseconds / 1000.0

            return None

        except Exception as e:
            # OCR failed
            return None


class ReferencePointSelector:
    """UI for selecting which point on the bounding box to track."""

    REFERENCE_POINTS = {
        "1": ("top-left", lambda x, y, w, h: (x, y)),
        "2": ("top-center", lambda x, y, w, h: (x + w // 2, y)),
        "3": ("top-right", lambda x, y, w, h: (x + w, y)),
        "4": ("center-left", lambda x, y, w, h: (x, y + h // 2)),
        "5": ("center", lambda x, y, w, h: (x + w // 2, y + h // 2)),
        "6": ("center-right", lambda x, y, w, h: (x + w, y + h // 2)),
        "7": ("bottom-left", lambda x, y, w, h: (x, y + h)),
        "8": ("bottom-center", lambda x, y, w, h: (x + w // 2, y + h)),
        "9": ("bottom-right", lambda x, y, w, h: (x + w, y + h)),
    }

    def __init__(self):
        self.selected_point = "bottom-center"  # Default
        self.selected_key = "8"

    def select_reference_point(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Tuple[str, callable]:
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
                cv2.putText(
                    display_frame,
                    key,
                    (pt_x + 10, pt_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            # Show selected point
            cv2.putText(
                display_frame,
                f"Selected: {self.selected_point} ({self.selected_key})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display_frame,
                "Press ENTER to confirm",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

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
            elif key == ord("r"):
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
                cv2.rectangle(
                    self.frame, self.start_point, self.end_point, (0, 255, 0), 2
                )
                cv2.putText(
                    self.frame,
                    "Release mouse to set selection",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

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
                cv2.putText(
                    self.frame,
                    "Press ENTER to confirm, 'r' to reset",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )


class ObjectTracker:
    """Tracks a selected object in a video and outputs position data."""

    def __init__(self, video_path: str, output_path: str = "output.mp4"):
        self.video_path = video_path
        self.output_path = output_path
        self.position_data = []
        self.calibration_data = None  # Dict with tick positions
        self.reference_length_cm = None
        self.reference_point_name = None
        self.reference_point_func = None
        self.timer_bbox = None  # Bounding box for timer region
        self.timer_rotation = 0  # Rotation angle for timer (0, 90, 180, 270)
        self.use_timer = False  # Whether to use OCR timer

    def calibrate_scale(self, reference_length_cm: float = 100.0) -> bool:
        """Calibrate the scale using a reference object (e.g., meter stick).

        Args:
            reference_length_cm: Length of reference object in centimeters

        Returns:
            True if calibration successful, False otherwise
        """
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError("Could not read video file")

        calibrator = MeterStickCalibrator()
        self.calibration_data = calibrator.calibrate(frame, reference_length_cm)
        self.reference_length_cm = reference_length_cm

        return self.calibration_data is not None

    def calibrate_timer(self) -> bool:
        """Calibrate the timer region for OCR timestamp extraction.

        Returns:
            True if timer calibration successful, False otherwise
        """
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError("Could not read video file")

        calibrator = TimerCalibrator()
        result = calibrator.calibrate(frame)

        if result is not None:
            self.timer_bbox, self.timer_rotation = result
            return True
        return False

    def _pixel_to_cm(self, px: float, py: float) -> Tuple[float, float]:
        """Convert pixel coordinates to cm using calibration data.

        Uses piecewise linear interpolation between tick marks to approximate lens distortion.
        Each segment between tick marks has its own local scale (cm/pixel ratio).

        Args:
            px, py: Pixel coordinates

        Returns:
            Tuple of (x_cm, y_cm)
        """
        if self.calibration_data is None:
            return None, None

        tick_positions = self.calibration_data["tick_positions"]

        # Find the line segment along the ruler closest to the point
        # For now, we'll use a simple projection onto the ruler line
        # and interpolate based on that

        # Calculate distances from point to each tick
        min_dist = float("inf")
        closest_segment = 0

        for i in range(len(tick_positions) - 1):
            x1, y1, cm1 = tick_positions[i]
            x2, y2, cm2 = tick_positions[i + 1]

            # Distance from point to line segment
            # Project point onto line
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue

            t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy

            dist = np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_segment = i

        # Interpolate along the closest segment
        x1, y1, cm1 = tick_positions[closest_segment]
        x2, y2, cm2 = tick_positions[min(closest_segment + 1, len(tick_positions) - 1)]

        dx = x2 - x1
        dy = y2 - y1
        segment_length_px = np.sqrt(dx**2 + dy**2)

        if segment_length_px == 0:
            return cm1, 0

        # Project point onto segment to find position along ruler
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

        # Interpolate cm value
        cm_along_ruler = cm1 + t * (cm2 - cm1)

        # For y-axis, use perpendicular distance
        # Calculate local scale at this position
        cm_per_pixel = abs(cm2 - cm1) / segment_length_px

        # Calculate perpendicular distance from ruler line
        perp_dist_px = min_dist

        # Convert to cm
        perp_dist_cm = perp_dist_px * cm_per_pixel

        return cm_along_ruler, perp_dist_cm

    def select_object(self) -> Optional[Tuple[int, int, int, int]]:
        """Allow user to select object with mouse by drawing a bounding box."""
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError("Could not read video file")

        selector = ObjectSelector()
        bbox = selector.select_roi(frame)

        return bbox

    def select_reference_point(self, bbox: Tuple[int, int, int, int]) -> bool:
        """Allow user to select which point on the bounding box to track.

        Args:
            bbox: Bounding box (x, y, w, h)

        Returns:
            True if selection successful, False otherwise
        """
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError("Could not read video file")

        selector = ReferencePointSelector()
        self.reference_point_name, self.reference_point_func = (
            selector.select_reference_point(frame, bbox)
        )

        return self.reference_point_func is not None

    def track_object(self, initial_bbox: Tuple[int, int, int, int]):
        """Track the object through the video and create output with bounding boxes."""
        # Initialize OpenCV tracker
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        # Read first frame and initialize tracker
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")

        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, initial_bbox)

        frame_number = 0
        timer_started = not self.use_timer  # If not using timer, consider it "started"
        skipped_frames = 0

        print(f"\nTracking object through {total_frames} frames...")
        if self.use_timer:
            print("Timer mode: Will skip frames until timer starts (non-zero)")
        print("Press 'q' to quit early")

        # Create window for tracking display
        cv2.namedWindow("Tracking Progress")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Check if timer has started (if using timer)
            if self.use_timer and not timer_started and self.timer_bbox is not None:
                tx, ty, tw, th = self.timer_bbox
                timer_roi = frame[ty:ty+th, tx:tx+tw]
                ocr_timestamp = TimerCalibrator._ocr_timer(timer_roi, self.timer_rotation)

                if ocr_timestamp is not None and ocr_timestamp > 0:
                    timer_started = True
                    print(f"\n✓ Timer started at frame {frame_number} (t={ocr_timestamp:.3f}s)")
                else:
                    # Skip this frame - timer hasn't started
                    skipped_frames += 1
                    frame_number += 1
                    if frame_number % 100 == 0:
                        print(f"Waiting for timer to start... (skipped {skipped_frames} frames)")
                    continue

            # Update tracker
            success, bbox = tracker.update(frame)

            if success:
                x, y, w, h = [int(v) for v in bbox]

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Calculate reference point position
                if self.reference_point_func is not None:
                    ref_x, ref_y = self.reference_point_func(x, y, w, h)
                else:
                    # Default to bottom-center
                    ref_x = x + w // 2
                    ref_y = y + h

                # Store position data
                timestamp = frame_number / fps if fps > 0 else 0
                data_entry = {
                    "frame": frame_number,
                    "timestamp": round(timestamp, 3),
                    "reference_point": self.reference_point_name or "bottom-center",
                    "position_x_pixels": ref_x,
                    "position_y_pixels": ref_y,
                    "bbox_pixels": {"x": x, "y": y, "w": w, "h": h},
                }

                # Extract OCR timestamp if timer tracking is enabled
                if self.use_timer and self.timer_bbox is not None:
                    tx, ty, tw, th = self.timer_bbox
                    timer_roi = frame[ty:ty+th, tx:tx+tw]
                    ocr_timestamp = TimerCalibrator._ocr_timer(timer_roi, self.timer_rotation)
                    if ocr_timestamp is not None:
                        data_entry["timestamp_ocr"] = round(ocr_timestamp, 3)

                # Add scaled measurements if calibration was performed
                if self.calibration_data is not None:
                    pos_x_cm, pos_y_cm = self._pixel_to_cm(ref_x, ref_y)
                    bbox_tl_x_cm, bbox_tl_y_cm = self._pixel_to_cm(x, y)
                    bbox_br_x_cm, bbox_br_y_cm = self._pixel_to_cm(x + w, y + h)

                    data_entry.update(
                        {
                            "position_x_cm": round(pos_x_cm, 3)
                            if pos_x_cm is not None
                            else None,
                            "position_y_cm": round(pos_y_cm, 3)
                            if pos_y_cm is not None
                            else None,
                            "bbox_cm": {
                                "x": round(bbox_tl_x_cm, 3)
                                if bbox_tl_x_cm is not None
                                else None,
                                "y": round(bbox_tl_y_cm, 3)
                                if bbox_tl_y_cm is not None
                                else None,
                                "w": round(bbox_br_x_cm - bbox_tl_x_cm, 3)
                                if bbox_br_x_cm is not None and bbox_tl_x_cm is not None
                                else None,
                                "h": round(bbox_br_y_cm - bbox_tl_y_cm, 3)
                                if bbox_br_y_cm is not None and bbox_tl_y_cm is not None
                                else None,
                            },
                        }
                    )

                self.position_data.append(data_entry)

                # Draw reference point
                cv2.circle(frame, (ref_x, ref_y), 8, (0, 0, 255), -1)
                cv2.circle(frame, (ref_x, ref_y), 3, (255, 255, 255), -1)

                # Draw timer region if enabled
                if self.use_timer and self.timer_bbox is not None:
                    tx, ty, tw, th = self.timer_bbox
                    cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (0, 255, 255), 2)

                # Add position text
                ref_name = self.reference_point_name or "bottom-center"
                if self.calibration_data is not None:
                    pos_x_cm, pos_y_cm = self._pixel_to_cm(ref_x, ref_y)
                    cv2.putText(
                        frame,
                        f"{ref_name}: ({ref_x}, {ref_y}) px = ({round(pos_x_cm, 1)}, {round(pos_y_cm, 1)}) cm",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        f"{ref_name}: ({ref_x}, {ref_y}) px",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                cv2.putText(
                    frame,
                    f"Frame: {frame_number}/{total_frames}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            else:
                # Tracking failed
                cv2.putText(
                    frame,
                    "Tracking lost!",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            # Draw progress bar at bottom of frame
            progress_pct = frame_number / total_frames if total_frames > 0 else 0
            bar_width = width - 20
            bar_height = 30
            bar_x = 10
            bar_y = height - bar_height - 10

            # Draw background (dark gray)
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (50, 50, 50),
                -1,
            )

            # Draw filled portion (green)
            filled_width = int(bar_width * progress_pct)
            if filled_width > 0:
                cv2.rectangle(
                    frame,
                    (bar_x, bar_y),
                    (bar_x + filled_width, bar_y + bar_height),
                    (0, 255, 0),
                    -1,
                )

            # Draw border
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (255, 255, 255),
                2,
            )

            # Draw percentage text
            progress_text = f"{progress_pct * 100:.1f}%"
            text_size = cv2.getTextSize(
                progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )[0]
            text_x = bar_x + (bar_width - text_size[0]) // 2
            text_y = bar_y + (bar_height + text_size[1]) // 2
            cv2.putText(
                frame,
                progress_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Display frame
            cv2.imshow("Tracking Progress", frame)

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nTracking cancelled by user")
                break

            # Write frame to output video
            out.write(frame)
            frame_number += 1

            # Progress indicator (console)
            if frame_number % 30 == 0:
                progress = (frame_number / total_frames) * 100
                print(
                    f"Progress: {progress:.1f}% ({frame_number}/{total_frames} frames)"
                )

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Tracking complete! Processed {frame_number} frames.")
        if self.use_timer and skipped_frames > 0:
            print(f"Skipped {skipped_frames} frames before timer started.")

    def save_position_data(self, output_file: str = "position_data.json"):
        """Save the position data to a JSON file."""
        output_data = {
            "metadata": {
                "video_path": self.video_path,
                "output_path": self.output_path,
                "total_frames": len(self.position_data),
                "calibrated": self.calibration_data is not None,
                "reference_point": self.reference_point_name,
                "timer_enabled": self.use_timer,
            },
            "tracking_data": self.position_data,
        }

        # Add calibration info if available
        if self.calibration_data is not None:
            output_data["metadata"]["calibration"] = {
                "reference_length_cm": self.reference_length_cm,
                "tick_positions": self.calibration_data["tick_positions"],
                "method": "piecewise linear interpolation with lens distortion correction",
            }

        # Add timer info if available
        if self.use_timer and self.timer_bbox is not None:
            output_data["metadata"]["timer"] = {
                "bbox": {
                    "x": self.timer_bbox[0],
                    "y": self.timer_bbox[1],
                    "w": self.timer_bbox[2],
                    "h": self.timer_bbox[3],
                },
                "rotation": self.timer_rotation,
                "method": "tesseract OCR",
            }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Position data saved to {output_file}")

    def run(self, calibrate: bool = True, reference_length_cm: float = 100.0, use_timer: bool = False):
        """Run the complete tracking pipeline.

        Args:
            calibrate: Whether to perform scale calibration
            reference_length_cm: Length of reference object in centimeters (default 100 for meter stick)
            use_timer: Whether to use OCR timer for timestamps
        """
        step = 1

        # Optional calibration step
        if calibrate:
            print(f"Step {step}: Calibrate scale with reference object")
            if not self.calibrate_scale(reference_length_cm):
                print("Calibration cancelled. Continuing without scale calibration.")
            step += 1

        # Optional timer calibration step
        if use_timer:
            print(f"Step {step}: Calibrate on-screen timer")
            if self.calibrate_timer():
                self.use_timer = True
            else:
                print("Timer calibration cancelled. Continuing without timer tracking.")
            step += 1

        print(f"Step {step}: Select object to track")
        initial_bbox = self.select_object()

        if initial_bbox is None:
            print("No object selected. Exiting.")
            return
        step += 1

        print(f"Step {step}: Select reference point on bounding box")
        if not self.select_reference_point(initial_bbox):
            print("Reference point selection failed. Exiting.")
            return
        step += 1

        print(f"Step {step}: Tracking object in video...")
        self.track_object(initial_bbox)
        step += 1

        print(f"Step {step}: Saving position data...")
        self.save_position_data()

        print(f"\n{'=' * 50}")
        print(f"✓ Done!")
        print(f"  Output video: {self.output_path}")
        print(f"  Total frames tracked: {len(self.position_data)}")
        print(f"  Reference point: {self.reference_point_name}")
        if self.calibration_data is not None:
            print(f"  Calibration: Piecewise linear (lens distortion corrected)")
            print(f"  Reference length: {self.reference_length_cm} cm")
        print(f"{'=' * 50}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Track objects in videos with precise position measurements"
    )
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument(
        "output_path", nargs="?", default="output.mp4", help="Path to output video file"
    )
    parser.add_argument(
        "--stick-length",
        type=float,
        default=100.0,
        help="Length of reference stick in centimeters (default: 100.0 for a meter stick)",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip calibration step (track in pixels only)",
    )
    parser.add_argument(
        "--use-timer",
        action="store_true",
        help="Extract timestamps from on-screen timer using OCR (requires pytesseract and Tesseract OCR)",
    )

    args = parser.parse_args()

    tracker = ObjectTracker(args.video_path, args.output_path)
    tracker.run(
        calibrate=not args.no_calibrate,
        reference_length_cm=args.stick_length,
        use_timer=args.use_timer,
    )


if __name__ == "__main__":
    main()
