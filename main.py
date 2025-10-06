import cv2
import numpy as np
from norfair import Detection, Tracker, Video
from typing import List, Tuple, Optional
import json


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

        print("=== Object Selection ===")
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

                # Calculate bottom-most pixel position
                bottom_y = y + h
                center_x = x + w // 2

                # Store position data
                timestamp = frame_number / fps if fps > 0 else 0
                self.position_data.append({
                    "frame": frame_number,
                    "timestamp": round(timestamp, 3),
                    "bottom_x": center_x,
                    "bottom_y": bottom_y,
                    "bbox": {"x": x, "y": y, "w": w, "h": h}
                })

                # Draw bottom point
                cv2.circle(frame, (center_x, bottom_y), 5, (0, 0, 255), -1)

                # Add position text
                cv2.putText(frame, f"Bottom: ({center_x}, {bottom_y})",
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
        with open(output_file, 'w') as f:
            json.dump(self.position_data, f, indent=2)
        print(f"Position data saved to {output_file}")

    def run(self):
        """Run the complete tracking pipeline."""
        print("Step 1: Select object to track")
        initial_bbox = self.select_object()

        if initial_bbox is None:
            print("No object selected. Exiting.")
            return

        print(f"Step 2: Tracking object in video...")
        self.track_object(initial_bbox)

        print(f"Step 3: Saving position data...")
        self.save_position_data()

        print(f"Done! Output video: {self.output_path}")
        print(f"Total frames tracked: {len(self.position_data)}")


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
