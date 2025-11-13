# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Videographer is a Python application for tracking objects in videos with precise position measurements. It uses OpenCV's CSRT tracker for frame-by-frame object tracking and provides interactive calibration for lens distortion correction using an adjustable meter stick interface.

## Development Commands

### Setup
```bash
# Install dependencies (recommended)
uv sync

# Or using pip
pip install opencv-contrib-python>=4.8.0 numpy>=1.24.0

# For timer OCR feature (optional)
pip install pytesseract>=0.3.10
# Also requires system installation: brew install tesseract (macOS) or apt-get install tesseract-ocr (Linux)
```

### Running the Application
```bash
# Basic tracking workflow
python main.py <input_video_path> [output_video_path] [options]

# Examples
python main.py input.mp4 tracked_output.mp4
python main.py input.mp4 tracked_output.mp4 --stick-length 30  # Use 30cm ruler
python main.py input.mp4 tracked_output.mp4 --no-calibrate     # Skip calibration
python main.py input.mp4 tracked_output.mp4 --use-timer        # Extract timestamps from on-screen timer
```

### Command-line Options
- `--stick-length LENGTH`: Length of reference stick in centimeters (default: 100.0)
- `--no-calibrate`: Skip calibration step (track in pixels only)
- `--use-timer`: Extract timestamps from on-screen timer using OCR (requires pytesseract)

### CSV Export
```bash
# Convert JSON position data to CSV
python to_csv.py position_data.json [options]

# Common options:
# -r/--reference: position, bbox_top, bbox_bottom, bbox_left, bbox_right, bbox_center_x, bbox_center_y
# -a/--axis: x or y
# -o/--offset: offset value in cm
# --header: include CSV header
# --delimiter: CSV delimiter (default comma)

# Example: Extract y-axis bottom of bbox with offset
python to_csv.py position_data.json -r bbox_bottom -a y -o 1.0 --header > output.csv
```

## Architecture

### Core Components

**TimerCalibrator** (main.py:365-636)
- Interactive UI for selecting on-screen timer region with bounding box
- **Rotation support** for vertical/rotated videos (0°, 90°, 180°, 270°)
- OCR-based timestamp extraction using Tesseract
- Supports formats: MM:SS.mmm, MM:SS, SS.mmm, SS
- Image preprocessing (rotation → grayscale → histogram equalization → thresholding) for better OCR
- Regex-based parsing to convert text to seconds
- **OCR validation with retry**: Tests OCR during calibration, shows raw text and parsed result
  - If parsing fails, prompts user to retry with adjusted region/rotation
  - Prevents proceeding with misconfigured timer
  - User can force continue, retry, or cancel
- Interactive rotation adjustment with keyboard shortcuts (0/9/1/2 keys)
- `_ocr_timer` method supports `return_raw` parameter for debugging

**ObjectTracker** (main.py:764-1252)
- Main orchestrator class that runs the complete tracking pipeline
- Manages calibration, timer calibration (optional), object selection, reference point selection, and tracking
- Converts pixel coordinates to centimeters using non-linear interpolation
- Extracts OCR timestamps from on-screen timer when enabled
- Outputs tracked video and position_data.json with both frame-based and OCR timestamps

**MeterStickCalibrator** (main.py:8-283)
- Interactive UI for two-step scale calibration:
  1. Mark meter stick endpoints (0cm and 100cm)
  2. Adjust tick marks (at 10cm intervals) to correct for lens distortion
- Stores tick positions as [x, y, cm_value] for piecewise linear interpolation
- Uses drag-and-drop interface for precise tick adjustment

**ReferencePointSelector** (main.py:285-363)
- UI for selecting which point on the bounding box to track
- 9 reference points: corners (4), edge midpoints (4), and center (1)
- Default: bottom-center (useful for tracking objects moving along a surface)
- Returns both point name and calculator function

**ObjectSelector** (main.py:366-453)
- Click-and-drag interface for drawing bounding box around object
- Minimum size validation (5x5 pixels)
- Reset capability for reselection

### Key Algorithms

**Lens Distortion Correction** (_pixel_to_cm method, main.py:489-560)
- Uses piecewise linear interpolation between adjustable tick marks
- For any pixel position:
  1. Finds closest ruler segment (between two tick marks)
  2. Projects point onto that segment
  3. Interpolates cm value based on local tick positions
  4. Calculates perpendicular distance using local scale (cm/pixel ratio)
- Handles fisheye, barrel, pincushion, and perspective distortion

**Tracking** (track_object method, main.py:954-1224)
- Uses OpenCV's CSRT (Channel and Spatial Reliability Tracking) tracker
- Initialized on user-selected bounding box in first frame
- Updates frame-by-frame with adaptive model
- Handles scale changes and partial occlusions
- **Auto-skip feature**: When timer is enabled, skips frames until timer shows non-zero value
- Stores both pixel and cm measurements for each frame
- Reports number of skipped frames at completion

### Data Format

**position_data.json Structure:**
```json
{
  "metadata": {
    "video_path": "...",
    "total_frames": 300,
    "calibrated": true,
    "reference_point": "bottom-center",
    "timer_enabled": true,
    "calibration": {
      "reference_length_cm": 100.0,
      "tick_positions": [[x, y, cm], ...],
      "method": "non-linear interpolation with lens distortion correction"
    },
    "timer": {
      "bbox": {"x": 100, "y": 50, "w": 200, "h": 60},
      "rotation": 90,
      "method": "tesseract OCR"
    }
  },
  "tracking_data": [
    {
      "frame": 0,
      "timestamp": 0.0,
      "timestamp_ocr": 0.123,
      "reference_point": "bottom-center",
      "position_x_pixels": 320,
      "position_y_pixels": 450,
      "position_x_cm": 42.5,
      "position_y_cm": 58.3,
      "bbox_pixels": {"x": ..., "y": ..., "w": ..., "h": ...},
      "bbox_cm": {"x": ..., "y": ..., "w": ..., "h": ...}
    }
  ]
}
```

### Coordinate System

- **Origin (0, 0)**: Start point of the meter stick (0cm mark)
- **X-axis**: Along the meter stick direction
- **Y-axis**: Perpendicular distance from the ruler
- All measurements in centimeters after calibration
- Pixel coordinates are also preserved for reference

## Important Notes

- The calibration tick adjustment is critical for accuracy with lens distortion
- CSRT tracker requires opencv-contrib-python (not standard opencv-python)
- Timer OCR feature requires pytesseract and system Tesseract OCR installation
- OCR accuracy varies with timer clarity, contrast, and video quality
- Output files (output.mp4, position_data.json) are gitignored
- The application uses interactive OpenCV windows - requires display capability

## Edge Cases and Limitations

### Tracking Failures
- When object goes off-frame, CSRT reports failure and no position data is recorded for those frames
- Frame numbers in JSON will have gaps where tracking failed
- Tracking typically does not auto-recover when object returns to frame

### Timer OCR
- Some frames may fail OCR, resulting in missing `timestamp_ocr` fields
- Very small, blurry, or low-contrast timers may not be readable
- OCR adds processing overhead to each frame
- Rotation is applied before OCR to handle vertical/rotated videos

### Timer Rotation for Vertical Videos
- Use cases: vertical videos recorded on phones that get rotated during processing
- The timer appears sideways (90°/270°) or upside-down (180°) in the rotated video
- During calibration, user selects rotation angle (0/9/1/2 keys for 0°/90°/180°/270°)
- Timer ROI is rotated before OCR preprocessing to ensure digits are upright
- Rotation metadata is stored in position_data.json for reference

### Timer Auto-Skip Feature
- When `--use-timer` is enabled, frames are automatically skipped until timer starts
- Detects timer start: first frame where OCR reads a non-zero timestamp
- Prevents tracking during setup/pre-experiment period
- Prints progress messages every 100 skipped frames
- Final report includes total number of skipped frames
- Use case: lab recordings where timer is visible but not started initially
