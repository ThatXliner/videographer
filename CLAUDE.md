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
```

### Running the Application
```bash
# Basic tracking workflow
python main.py <input_video_path> [output_video_path] [options]

# Examples
python main.py input.mp4 tracked_output.mp4
python main.py input.mp4 tracked_output.mp4 --stick-length 30  # Use 30cm ruler
python main.py input.mp4 tracked_output.mp4 --no-calibrate     # Skip calibration
```

### Command-line Options
- `--stick-length LENGTH`: Length of reference stick in centimeters (default: 100.0)
- `--no-calibrate`: Skip calibration step (track in pixels only)

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

### CSV Data Cleaning
```bash
# Clean CSV data by removing timestamp outliers and averaging duplicate timestamps
python clean_csv.py data.csv [options]

# Common options:
# -o/--output: output file path (default: stdout)
# --iqr-factor: IQR multiplier for timestamp outlier detection (default: 1.5, lower = more aggressive)
# --header: include CSV header in output
# --delimiter: CSV delimiter (default comma)

# Examples:
# Basic cleaning to stdout
python clean_csv.py raw_data.csv

# Save cleaned data to file
python clean_csv.py raw_data.csv -o cleaned_data.csv --header

# More aggressive timestamp outlier removal
python clean_csv.py raw_data.csv --iqr-factor 1.0 -o cleaned_data.csv

# Use with use-timer data (common workflow)
python to_csv.py position_data.json -r bbox_bottom -a y --header | python clean_csv.py /dev/stdin -o cleaned.csv

# Note: Only timestamps are filtered for outliers (e.g., 2.407 vs 0.1)
# Position values are never filtered - they are only averaged when timestamps match
```

## Architecture

### Core Components

**ObjectTracker** (main.py:455-778)
- Main orchestrator class that runs the complete tracking pipeline
- Manages calibration, object selection, reference point selection, and tracking
- Converts pixel coordinates to centimeters using non-linear interpolation
- Outputs tracked video and position_data.json

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

**Tracking** (track_object method, main.py:597-705)
- Uses OpenCV's CSRT (Channel and Spatial Reliability Tracking) tracker
- Initialized on user-selected bounding box in first frame
- Updates frame-by-frame with adaptive model
- Handles scale changes and partial occlusions
- Stores both pixel and cm measurements for each frame

### Data Format

**position_data.json Structure:**
```json
{
  "metadata": {
    "video_path": "...",
    "total_frames": 300,
    "calibrated": true,
    "reference_point": "bottom-center",
    "calibration": {
      "reference_length_cm": 100.0,
      "tick_positions": [[x, y, cm], ...],
      "method": "non-linear interpolation with lens distortion correction"
    }
  },
  "tracking_data": [
    {
      "frame": 0,
      "timestamp": 0.0,
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
- Output files (output.mp4, position_data.json) are gitignored
- The application uses interactive OpenCV windows - requires display capability
