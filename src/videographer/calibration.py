"""Scale calibration with adjustable tick marks for lens distortion correction."""

import cv2
import numpy as np
from typing import Optional


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

    def calibrate(self, frame: np.ndarray, stick_length_cm: float = 100.0) -> Optional[dict]:
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

        cv2.namedWindow("Calibrate Scale - Mark Meter Stick")
        cv2.setMouseCallback("Calibrate Scale - Mark Meter Stick", self._mouse_callback)

        print("\n=== Scale Calibration - Step 1 ===")
        print(f"Reference length: {stick_length_cm} cm")
        print("1. Click at one end of the meter stick")
        print("2. Click at the other end of the meter stick")
        print("3. Press ENTER to continue")
        print("4. Press 'r' to reset")
        print("5. Press ESC to cancel")

        while True:
            cv2.imshow("Calibrate Scale - Mark Meter Stick", self.frame)
            key = cv2.waitKey(1) & 0xFF

            # Enter key - confirm selection
            if key == 13 and self.start_point is not None and self.end_point is not None:
                break
            # ESC key - cancel
            elif key == 27:
                cv2.destroyAllWindows()
                return None
            # 'r' key - reset
            elif key == ord('r'):
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
            'tick_positions': self.tick_positions,
            'start_point': self.start_point,
            'end_point': self.end_point,
            'stick_length_cm': stick_length_cm
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
                cv2.putText(self.frame, "Click at the other end",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
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
                cv2.putText(temp_frame, f"{distance:.1f} pixels",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

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

        # Draw tick marks every 10cm (assuming 100cm stick)
        num_ticks = 11  # 0, 10, 20, ..., 100 cm
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
                cv2.putText(self.frame, label,
                           (tick_x + 5, tick_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Draw distance text
        mid_x = (self.start_point[0] + self.end_point[0]) // 2
        mid_y = (self.start_point[1] + self.end_point[1]) // 2
        cv2.putText(self.frame, f"{distance:.1f} pixels",
                   (mid_x + 10, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(self.frame, "Press ENTER to adjust ticks, 'r' to reset",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def _initialize_tick_positions(self):
        """Initialize tick positions along the line."""
        dx = self.end_point[0] - self.start_point[0]
        dy = self.end_point[1] - self.start_point[1]

        num_ticks = 11  # 0, 10, 20, ..., 100 cm
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

        cv2.setMouseCallback("Calibrate Scale - Mark Meter Stick", self._tick_adjust_callback)

        # Initial draw
        self._draw_adjustable_ticks()
        cv2.imshow("Calibrate Scale - Mark Meter Stick", self.frame)
        self.needs_redraw = False

        while True:
            # Only redraw if something changed
            if self.needs_redraw:
                self._draw_adjustable_ticks()
                cv2.imshow("Calibrate Scale - Mark Meter Stick", self.frame)
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
                handle_color = (255, 0, 255) if self.selected_tick == i else (0, 255, 255)
                cv2.circle(self.frame, (tick_x, tick_y), 6, handle_color, -1)

                # Draw label for major ticks
                if i % 2 == 0:
                    label = f"{int(cm_value)}"
                    cv2.putText(self.frame, label,
                               (tick_x + 5, tick_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        cv2.putText(self.frame, "Drag tick marks to adjust for lens distortion. Press ENTER when done.",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def _tick_adjust_callback(self, event, x, y, flags, param):
        """Handle mouse events for tick adjustment."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Find closest tick mark
            min_dist = float('inf')
            closest_tick = None
            for i, (tick_x, tick_y, _) in enumerate(self.tick_positions):
                dist = np.sqrt((x - tick_x)**2 + (y - tick_y)**2)
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
