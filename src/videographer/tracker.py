"""Object tracking with position data export."""

import json
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from .calibration import MeterStickCalibrator
from .selectors import ObjectSelector, ReferencePointSelector


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
        print(f"\nTracking object through {total_frames} frames...")
        print("Press 'q' to quit early")

        # Create window for tracking display
        cv2.namedWindow("Tracking Progress")

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
                    "bbox_pixels": {"x": x, "y": y, "w": w, "h": h},
                }

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

    def save_position_data(self, output_file: str = "position_data.json"):
        """Save the position data to a JSON file."""
        output_data = {
            "metadata": {
                "video_path": self.video_path,
                "output_path": self.output_path,
                "total_frames": len(self.position_data),
                "calibrated": self.calibration_data is not None,
                "reference_point": self.reference_point_name,
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

        with open(output_file, "w") as f:
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

        print(f"\n{'=' * 50}")
        print(f"✓ Done!")
        print(f"  Output video: {self.output_path}")
        print(f"  Total frames tracked: {len(self.position_data)}")
        print(f"  Reference point: {self.reference_point_name}")
        if self.calibration_data is not None:
            print(f"  Calibration: Piecewise linear (lens distortion corrected)")
            print(f"  Reference length: {self.reference_length_cm} cm")
        print(f"{'=' * 50}")
