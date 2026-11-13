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
# --delimiter: CSV delimiter (default tab)

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
python to_csv.py position_data.json -r bbox_bottom -a y --header --delimiter , | python clean_csv.py /dev/stdin -o cleaned.csv

# Note: Only timestamps are filtered for outliers (e.g., 2.407 vs 0.1)
# Position values are never filtered - they are only averaged when timestamps match
```

### Linear Region Extraction
```bash
# Extract linear portion of data by trimming start and end
python clean_for_linearized.py data.csv [options]

# Common options:
# -o/--output: output file path (default: stdout)
# --min-r2: minimum R² threshold for linearity (default: 0.8)
# --min-points: minimum number of points to keep (default: 3)
# --header: include CSV header in output
# --delimiter: CSV delimiter (default comma)

# Examples:
# Basic usage - extract region with R² >= 0.8
python clean_for_linearized.py data.csv --header

# More strict linearity requirement
python clean_for_linearized.py data.csv --min-r2 0.95 -o linear.csv --header

# Very strict - nearly perfect linearity
python clean_for_linearized.py data.csv --min-r2 0.99 --header

# Combined workflow: clean outliers, then extract linear region
python clean_csv.py raw_data.csv --header | python clean_for_linearized.py /dev/stdin --min-r2 0.9 -o linear.csv --header

# Use case: Extract constant-velocity portion from motion data
# - Removes acceleration phase at start
# - Removes deceleration phase at end
# - Keeps only the linear (constant velocity) middle section
```

## Web app

`web/` is a static, build-free browser version (vanilla JS ES modules) deployed to
https://bryanhu.com/videographer/ by `.github/workflows/pages.yml` on pushes to `main`.

- `app.js`: UI state machine. A phase table maps each step (`load`, `calibrateEndpoints`, `calibrateTicks`, `selectObject`, `selectReference`, `tracking`, `results`) to its pointer handlers, render, and Enter/Esc/R actions
- `video.js`: frame source. Hidden `<video>`, fps detection, `frameAt(i, fps)` seeks to `(i + 0.5) / fps`. Tracking decodes one frame ahead of the worker
- `tracker.js`: main-thread client for `tracker-worker.js`. Starts the worker on page load and transfers each frame's pixels to it
- `tracker-worker.js`: classic worker that loads OpenCV.js from jsDelivr and runs `TrackerMIL` (CSRT isn't in any prebuilt OpenCV.js). It feeds MIL the blue channel, which is what the CLI's BGR frames give it
- `calibration.js`, `reference-points.js`, `export.js`: pure ports of `_pixel_to_cm`, `REFERENCE_POINTS`, and the `position_data.json` schema (including Python's `round` and frame numbering)
- `plot.js`: SVG position plot for the results step

```bash
# Tests (Node 22+, no dependencies)
node --test "web/tests/*.test.js"

# Regenerate the Python parity fixtures after changing main.py's math
uv run python web/tests/gen_fixtures.py

# Serve locally
python -m http.server -d web
```

## Architecture

### Core Components

**ObjectTracker** (main.py)
- Main orchestrator class that runs the complete tracking pipeline
- Manages calibration, object selection, reference point selection, and tracking
- Converts pixel coordinates to centimeters using non-linear interpolation
- Outputs tracked video and position_data.json

**MeterStickCalibrator** (main.py)
- Interactive UI for two-step scale calibration:
  1. Mark meter stick endpoints (0cm and 100cm)
  2. Adjust tick marks (at 10cm intervals) to correct for lens distortion
- Stores tick positions as [x, y, cm_value] for piecewise linear interpolation
- Uses drag-and-drop interface for precise tick adjustment

**ReferencePointSelector** (main.py)
- UI for selecting which point on the bounding box to track
- 9 reference points: corners (4), edge midpoints (4), and center (1)
- Default: bottom-center (useful for tracking objects moving along a surface)
- Returns both point name and calculator function

**ObjectSelector** (main.py)
- Click-and-drag interface for drawing bounding box around object
- Minimum size validation (5x5 pixels)
- Reset capability for reselection

### Key Algorithms

**Lens Distortion Correction** (`ObjectTracker._pixel_to_cm`)
- Uses piecewise linear interpolation between adjustable tick marks
- For any pixel position:
  1. Finds closest ruler segment (between two tick marks)
  2. Projects point onto that segment
  3. Interpolates cm value based on local tick positions
  4. Calculates perpendicular distance using local scale (cm/pixel ratio)
- Handles fisheye, barrel, pincushion, and perspective distortion

**Tracking** (`ObjectTracker.track_object`)
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
