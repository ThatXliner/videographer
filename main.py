import cv2
import numpy as np
from norfair import Detection, Tracker, Video
from typing import List, Tuple, Optional
import json


class MeterStickCalibrator:
    """UI for calibrating scale using a meter stick."""

    def __init__(self):
        self.start_point = None
        self.end_point = None
        self.selecting = False
        self.frame = None
        self.clone = None
        self.pixel_distance = None

    def calibrate(self, frame: np.ndarray, stick_length_cm: float = 100.0) -> Optional[float]:
        """Display UI for user to mark the meter stick endpoints.

        Args:
            frame: Video frame containing the meter stick
            stick_length_cm: Length of the reference stick in centimeters (default 100 for meter stick)

        Returns:
            Scale factor in cm/pixel, or None if cancelled
        """
        self.frame = frame.copy()
        self.clone = frame.copy()
        self.start_point = None
        self.end_point = None
        self.selecting = False

        cv2.namedWindow("Calibrate Scale - Mark Meter Stick")
        cv2.setMouseCallback("Calibrate Scale - Mark Meter Stick", self._mouse_callback)

        print("\n=== Scale Calibration ===")
        print(f"Reference length: {stick_length_cm} cm")
        print("1. Click at one end of the meter stick")
        print("2. Click at the other end of the meter stick")
        print("3. Press ENTER to confirm")
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

        cv2.destroyAllWindows()

        # Calculate pixel distance
        if self.start_point and self.end_point:
            dx = self.end_point[0] - self.start_point[0]
            dy = self.end_point[1] - self.start_point[1]
            pixel_distance = np.sqrt(dx**2 + dy**2)

            # Calculate scale: cm per pixel
            scale = stick_length_cm / pixel_distance

            print(f"\n✓ Calibration complete!")
            print(f"  Pixel distance: {pixel_distance:.2f} pixels")
            print(f"  Scale: {scale:.4f} cm/pixel")
            print(f"  Scale: {1/scale:.2f} pixels/cm")

            return scale

        return None

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
        cv2.putText(self.frame, "Press ENTER to confirm, 'r' to reset",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


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

    def select_reference_point(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[str, callable]:
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


class ObjectTracker:
    """Tracks a selected object in a video and outputs position data."""

    def __init__(self, video_path: str, output_path: str = "output.mp4"):
        self.video_path = video_path
        self.output_path = output_path
        self.position_data = []
        self.scale = None  # cm per pixel
        self.reference_length_cm = None
        self.reference_point_name = None
        self.reference_point_func = None

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
        self.scale = calibrator.calibrate(frame, reference_length_cm)
        self.reference_length_cm = reference_length_cm

        return self.scale is not None

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
        self.reference_point_name, self.reference_point_func = selector.select_reference_point(frame, bbox)

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
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        # Read first frame and initialize tracker
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")

        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, initial_bbox)

        frame_number = 0
        print(f"\nTracking object through {total_frames} frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

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
                    "bbox_pixels": {"x": x, "y": y, "w": w, "h": h}
                }

                # Add scaled measurements if calibration was performed
                if self.scale is not None:
                    pos_x_cm = ref_x * self.scale
                    pos_y_cm = ref_y * self.scale
                    data_entry.update({
                        "position_x_cm": round(pos_x_cm, 3),
                        "position_y_cm": round(pos_y_cm, 3),
                        "bbox_cm": {
                            "x": round(x * self.scale, 3),
                            "y": round(y * self.scale, 3),
                            "w": round(w * self.scale, 3),
                            "h": round(h * self.scale, 3)
                        }
                    })

                self.position_data.append(data_entry)

                # Draw reference point
                cv2.circle(frame, (ref_x, ref_y), 8, (0, 0, 255), -1)
                cv2.circle(frame, (ref_x, ref_y), 3, (255, 255, 255), -1)

                # Add position text
                ref_name = self.reference_point_name or "bottom-center"
                if self.scale is not None:
                    cv2.putText(frame, f"{ref_name}: ({ref_x}, {ref_y}) px = ({round(ref_x * self.scale, 1)}, {round(ref_y * self.scale, 1)}) cm",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"{ref_name}: ({ref_x}, {ref_y}) px",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(frame, f"Frame: {frame_number}/{total_frames}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Tracking failed
                cv2.putText(frame, "Tracking lost!",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Write frame to output video
            out.write(frame)
            frame_number += 1

            # Progress indicator
            if frame_number % 30 == 0:
                progress = (frame_number / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_number}/{total_frames} frames)")

        cap.release()
        out.release()
        print(f"Tracking complete! Processed {frame_number} frames.")

    def save_position_data(self, output_file: str = "position_data.json"):
        """Save the position data to a JSON file."""
        output_data = {
            "metadata": {
                "video_path": self.video_path,
                "output_path": self.output_path,
                "total_frames": len(self.position_data),
                "calibrated": self.scale is not None
            },
            "tracking_data": self.position_data
        }

        # Add calibration info if available
        if self.scale is not None:
            output_data["metadata"]["calibration"] = {
                "scale_cm_per_pixel": round(self.scale, 6),
                "scale_pixels_per_cm": round(1/self.scale, 6),
                "reference_length_cm": self.reference_length_cm
            }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Position data saved to {output_file}")

    def run(self, calibrate: bool = True, reference_length_cm: float = 100.0):
        """Run the complete tracking pipeline.

        Args:
            calibrate: Whether to perform scale calibration
            reference_length_cm: Length of reference object in centimeters (default 100 for meter stick)
        """
        step = 1

        # Optional calibration step
        if calibrate:
            print(f"Step {step}: Calibrate scale with reference object")
            if not self.calibrate_scale(reference_length_cm):
                print("Calibration cancelled. Continuing without scale calibration.")
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

        print(f"\n{'='*50}")
        print(f"✓ Done!")
        print(f"  Output video: {self.output_path}")
        print(f"  Total frames tracked: {len(self.position_data)}")
        print(f"  Reference point: {self.reference_point_name}")
        if self.scale is not None:
            print(f"  Scale: {self.scale:.4f} cm/pixel")
        print(f"{'='*50}")


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path> [output_path]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"

    tracker = ObjectTracker(video_path, output_path)
    tracker.run()


if __name__ == "__main__":
    main()
