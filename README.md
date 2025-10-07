# VideoGrapher

> Note that this app so far is 100% written by AI.

A Python application for tracking objects in videos with precise position measurements. Features interactive calibration for lens distortion correction, customizable reference points, and exportable position data.

## Citation

If you use this software in your research or project, please cite it as:

**BibTeX:**
```bibtex
@software{hu2025videographer,
  author = {Hu, Bryan},
  title = {VideoGrapher: Object Tracking with Lens Distortion Correction},
  year = {2025},
  url = {https://github.com/ThatXliner/videographer},
  version = {0.1.0}
}
```

**APA:**
```
Hu, B. (2025). VideoGrapher: Object Tracking with Lens Distortion Correction (Version 0.1.0) [Computer software]. https://github.com/ThatXliner/videographer
```

**MLA:**
```
Hu, Bryan. VideoGrapher: Object Tracking with Lens Distortion Correction. Version 0.1.0, 2025, https://github.com/ThatXliner/videographer.
```

See also: [CITATION.cff](CITATION.cff)

## Features

### 🎯 Object Tracking
- Interactive object selection via mouse-drawn bounding box
- Frame-by-frame tracking using OpenCV CSRT tracker
- 9 selectable reference points on the bounding box (corners, edges, center)
- Visual output video with tracking overlay

### 📏 Scale Calibration
- Interactive meter stick calibration with real-world measurements
- **Adjustable tick marks** for lens distortion correction
- Piecewise linear interpolation between tick marks for accurate measurements
- Visual tick mark display (every 10cm with labels)

### 📊 Data Export
- JSON export with comprehensive position data
- CSV converter with flexible options
- Both pixel and centimeter measurements
- Frame-by-frame timestamp tracking

## Installation

### Using `uv` (recommended)
```bash
uv sync
```

### Using `pip`
```bash
pip install opencv-contrib-python>=4.8.0 norfair>=2.2.0 numpy>=1.24.0
```

## Usage

### Basic Tracking

```bash
# Using uv (recommended)
uv run videographer <input_video_path> [options]

# Or with python -m
python -m videographer <input_video_path> [options]

# After installation (uv sync)
videographer <input_video_path> [options]
```

**Examples:**
```bash
# Note that all instances of `videographer` are interchangeable with `uv run videographer` or `python -m videographer`
# Basic usage with default settings
videographer input.mp4

# Custom output path
videographer input.mp4 -o tracked_output.mp4

# Skip calibration (pixels only)
videographer input.mp4 --no-calibration

# Custom reference length (e.g., 50cm ruler)
videographer input.mp4 --reference-length 50

# Custom data output file
videographer input.mp4 --output-data my_data.json

# Get help
videographer --help
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `video_path` | Input video file (required) | - |
| `-o, --output` | Output video file path | `output.mp4` |
| `--no-calibration` | Skip scale calibration (pixels only) | Off |
| `--reference-length` | Reference object length in cm | `100.0` |
| `--output-data` | Output JSON data file path | `position_data.json` |
| `-v, --version` | Show version | - |
| `-h, --help` | Show help message | - |

### Workflow

The application guides you through an interactive workflow:

#### **Step 1: Scale Calibration**

Mark the meter stick endpoints:
1. Click at the **0cm** end of the meter stick
2. Click at the **100cm** end
3. Press **ENTER** to continue

Adjust tick marks for lens distortion:
1. Tick marks appear at 0, 10, 20, ..., 100 cm
2. **Click and drag** any tick mark to adjust its position
3. Align ticks with the actual ruler markings in your video
4. Press **ENTER** when satisfied

![Calibration UI](docs/calibration.png)

> **Tip:** This step corrects for lens distortion. If your video has fisheye or other distortion, adjust the outer tick marks to match the curved ruler.

#### **Step 2: Select Object**

1. Click and drag to draw a bounding box around your object
2. Press **ENTER** to confirm
3. Press **'r'** to reset if needed

#### **Step 3: Choose Reference Point**

Select which point on the bounding box to track:

```
1: Top-Left      2: Top-Center      3: Top-Right
4: Center-Left   5: Center          6: Center-Right
7: Bottom-Left   8: Bottom-Center   9: Bottom-Right
```

- Press the number key (1-9) to select
- Default is **8 (Bottom-Center)**
- Press **ENTER** to confirm

#### **Step 4: Track Object**

The application processes the video automatically:
- Progress updates every 30 frames
- Creates output video with tracking overlay
- Generates `position_data.json`

### Output Files

#### **Output Video**
- Bounding box (green rectangle)
- Reference point marker (red circle with white center)
- Position overlay showing pixel and cm coordinates
- Frame counter

#### **position_data.json**

Comprehensive tracking data with metadata:

```json
{
  "metadata": {
    "video_path": "input.mp4",
    "total_frames": 300,
    "calibrated": true,
    "reference_point": "bottom-center",
    "calibration": {
      "reference_length_cm": 100.0,
      "tick_positions": [[x1, y1, 0], [x2, y2, 10], ...],
      "method": "piecewise linear interpolation with lens distortion correction"
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
      "bbox_pixels": {"x": 280, "y": 400, "w": 80, "h": 50},
      "bbox_cm": {"x": 37.2, "y": 51.8, "w": 10.6, "h": 6.5}
    },
    ...
  ]
}
```

**Data Fields:**
- `frame`: Frame number (0-indexed)
- `timestamp`: Time in seconds
- `reference_point`: Which point was tracked
- `position_x/y_pixels`: Pixel coordinates of reference point
- `position_x/y_cm`: Real-world coordinates in centimeters
- `bbox_pixels/cm`: Bounding box in both units

## CSV Export

Convert JSON data to CSV with flexible options:

```bash
python to_csv.py position_data.json [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-r, --reference` | Reference point to extract | `position` |
| `-a, --axis` | Axis to extract (x or y) | `x` |
| `-o, --offset` | Offset to subtract (cm) | `0.0` |
| `--header` | Include CSV header | Off |
| `--delimiter` | CSV delimiter | `,` |

### Reference Point Options

- `position` - Use the tracked reference point (default)
- `bbox_top` - Top edge of bounding box
- `bbox_bottom` - Bottom edge of bounding box
- `bbox_left` - Left edge of bounding box
- `bbox_right` - Right edge of bounding box
- `bbox_center_x` - Horizontal center of bounding box
- `bbox_center_y` - Vertical center of bounding box

### Examples

**Basic usage:**
```bash
python to_csv.py position_data.json
```

**Extract bottom of bounding box with 1cm offset:**
```bash
python to_csv.py position_data.json -r bbox_bottom -a y -o 1.0
```

**With header, save to file:**
```bash
python to_csv.py position_data.json --header > output.csv
```

**Tab-delimited output:**
```bash
python to_csv.py position_data.json --delimiter $'\t' > output.tsv
```

**Get help:**
```bash
python to_csv.py --help
```

## Advanced Features

### Lens Distortion Correction

The adjustable tick marks allow you to correct for lens distortion:

1. After marking the meter stick endpoints, tick marks appear
2. Drag each tick to align with the actual ruler markings
3. The system uses **piecewise linear interpolation** between ticks
4. Each position is converted using the local scale at that point

This provides accurate measurements even with:
- Fisheye lenses
- Barrel/pincushion distortion
- Non-uniform scaling across the frame

### How Distortion Correction Works

The calibration system approximates lens distortion using **piecewise linear interpolation**:

1. Stores adjusted positions for all 11 tick marks (0-100cm)
2. For each tracked point, finds the closest ruler segment (between two ticks)
3. Projects the point onto that segment
4. Uses **linear interpolation** within that segment based on local tick positions
5. Calculates perpendicular distance using the local scale (cm/pixel ratio)

**What this means:** The ruler is divided into 10 segments (0-10cm, 10-20cm, etc.). Each segment can have its own scale, allowing the system to approximate curved distortion by chaining together straight segments. More tick marks = better approximation of smooth nonlinear distortion.

**Limitations:** This works well for moderate barrel/pincushion/fisheye distortion, but cannot capture rapid distortion changes within a 10cm segment. For extreme distortion, you'd need more tick marks or a parametric lens model.

### Customizable Reference Points

Choose the most appropriate tracking point for your analysis:
- **Bottom-center**: Good for tracking objects moving along a surface
- **Center**: Good for projectile motion or general tracking
- **Top/Bottom edges**: For measuring object heights
- **Left/Right edges**: For measuring widths or leading/trailing edges

## Tips for Best Results

### Camera Setup
- Mount camera on stable tripod
- Keep meter stick parallel to the plane of motion
- Ensure good lighting and contrast
- Avoid motion blur (use faster shutter speed)

### Calibration
- Use a high-contrast meter stick that's clearly visible
- Place the meter stick in the same plane as object motion
- If using adjustable ticks, zoom in to align precisely
- Take time to adjust distorted tick marks carefully

### Object Selection
- Draw bounding box tightly around the object
- Include distinctive features for better tracking
- Choose a reference point that's consistently visible
- Test tracking on a few frames first

### Troubleshooting
- **Tracking lost**: Object may be too small or moving too fast
- **Inaccurate measurements**: Check tick mark alignment
- **Jittery tracking**: Try smaller bounding box or different tracker settings

## Technical Details

### Dependencies
- **opencv-contrib-python**: CSRT tracker and video I/O
- **norfair**: Object tracking framework (currently used minimally)
- **numpy**: Numerical operations

### Tracking Algorithm
- Uses OpenCV's CSRT (Channel and Spatial Reliability Tracking) tracker
- Initializes on user-selected bounding box
- Updates frame-by-frame with adaptive model
- Handles scale changes and partial occlusions

### Coordinate System
- Origin (0, 0) is at the **start point** of the meter stick
- X-axis runs along the meter stick direction
- Y-axis is perpendicular distance from the ruler
- All measurements in centimeters after calibration

## Project Structure

```
videographer/
├── videographer/        # Main package
│   ├── __init__.py     # Package exports
│   ├── __main__.py     # CLI entry point
│   ├── calibration.py  # Scale calibration with tick marks
│   ├── selectors.py    # Object and reference point selectors
│   └── tracker.py      # Object tracking and data export
├── to_csv.py           # CSV export tool
├── position_data.json  # Generated tracking data (gitignored)
├── output.mp4          # Generated output video (gitignored)
├── pyproject.toml      # Project dependencies and scripts
└── README.md           # This file
```

## Contributing

Suggestions and improvements welcome! Key areas for enhancement:
- Additional tracker algorithms
- Automatic object detection
- Multi-object tracking
- Real-time preview during tracking
- GUI application

## License

MIT License - feel free to use for research, education, or commercial projects.

## Acknowledgments

- OpenCV for computer vision tools
- Norfair for tracking framework
- CSRT tracker algorithm by Lukežič et al.
