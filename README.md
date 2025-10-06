# Videographer

A Python application for tracking objects in videos frame by frame using the Norfair library.

## Features

- Interactive object selection via mouse-drawn bounding box
- Frame-by-frame object tracking using OpenCV CSRT tracker and Norfair
- Visual output video with tracking bounding boxes
- Bottom-most pixel position tracking over time
- JSON export of position data

## Installation

```bash
uv sync
```

## Usage

```bash
python main.py <input_video_path> [output_video_path]
```

Example:
```bash
python main.py input.mp4 tracked_output.mp4
```

### How it works:

1. The first frame of the video will be displayed
2. Click and drag to draw a bounding box around the object you want to track
3. Press ENTER to confirm selection
4. The tracker will process the video and create:
   - An output video with bounding boxes drawn around the tracked object
   - A `position_data.json` file containing frame-by-frame position data

### Output Data

The `position_data.json` file contains an array of tracking data for each frame:

```json
[
  {
    "frame": 0,
    "timestamp": 0.0,
    "bottom_x": 320,
    "bottom_y": 450,
    "bbox": {"x": 280, "y": 400, "w": 80, "h": 50}
  },
  ...
]
```

- `frame`: Frame number
- `timestamp`: Time in seconds
- `bottom_x`: X coordinate of the bottom center point
- `bottom_y`: Y coordinate of the bottom-most pixel
- `bbox`: Bounding box coordinates (x, y, width, height)
