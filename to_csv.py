#!/usr/bin/env python3
"""
Convert position_data.json to CSV format with configurable options.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def extract_position(
    data_point: dict, reference: str, axis: str, offset: float = 0.0
) -> Optional[float]:
    """Extract position value from data point.

    Args:
        data_point: Single tracking data entry
        reference: Which value to extract ('position', 'bbox_top', 'bbox_bottom', 'bbox_left',
                   'bbox_right', 'bbox_center_x', 'bbox_center_y')
        axis: 'x' or 'y'
        offset: Offset to apply to the value

    Returns:
        Position value in cm, or None if not available
    """
    if reference == "position":
        # Use the tracked reference point
        key = f"position_{axis}_cm"
        value = data_point.get(key)
    elif reference.startswith("bbox_"):
        # Extract from bounding box
        bbox = data_point.get("bbox_cm")
        if bbox is None:
            return None

        if reference == "bbox_top":
            value = bbox.get("y")
        elif reference == "bbox_bottom":
            value = bbox.get("y")
            if value is not None and bbox.get("h") is not None:
                value = value + bbox.get("h")
        elif reference == "bbox_left":
            value = bbox.get("x")
        elif reference == "bbox_right":
            value = bbox.get("x")
            if value is not None and bbox.get("w") is not None:
                value = value + bbox.get("w")
        elif reference == "bbox_center_x":
            x = bbox.get("x")
            w = bbox.get("w")
            if x is not None and w is not None:
                value = x + w / 2
            else:
                value = None
        elif reference == "bbox_center_y":
            y = bbox.get("y")
            h = bbox.get("h")
            if y is not None and h is not None:
                value = y + h / 2
            else:
                value = None
        else:
            value = None
    else:
        value = None

    if value is not None:
        return value - offset
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert position_data.json to CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - extract position data with default settings
  python to_csv.py position_data.json

  # Extract bottom of bounding box with 1cm offset
  python to_csv.py position_data.json -r bbox_bottom -a y -o 1.0

  # Extract horizontal position from left edge of bbox
  python to_csv.py position_data.json -r bbox_left -a x

  # Save to file
  python to_csv.py position_data.json -r position -a x > output.csv

Reference options:
  position        - Use the tracked reference point (default)
  bbox_top        - Top edge of bounding box
  bbox_bottom     - Bottom edge of bounding box
  bbox_left       - Left edge of bounding box
  bbox_right      - Right edge of bounding box
  bbox_center_x   - Horizontal center of bounding box
  bbox_center_y   - Vertical center of bounding box
        """,
    )

    parser.add_argument("input_file", type=Path, help="Path to position_data.json file")

    parser.add_argument(
        "-r",
        "--reference",
        choices=[
            "position",
            "bbox_top",
            "bbox_bottom",
            "bbox_left",
            "bbox_right",
            "bbox_center_x",
            "bbox_center_y",
        ],
        default="position",
        help="Which reference point to extract (default: position)",
    )

    parser.add_argument(
        "-a",
        "--axis",
        choices=["x", "y"],
        default="x",
        help="Which axis to extract: x or y (default: x)",
    )

    parser.add_argument(
        "-o",
        "--offset",
        type=float,
        default=0.0,
        help="Offset to subtract from position values in cm (default: 0.0)",
    )

    parser.add_argument("--header", action="store_true", help="Include CSV header row")

    parser.add_argument(
        "--delimiter", default="\t", help="CSV delimiter (default: tab)"
    )

    args = parser.parse_args()

    # Load data
    if not args.input_file.exists():
        print(f"Error: File not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(args.input_file.read_text())
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}", file=sys.stderr)
        sys.exit(1)

    # Print header if requested
    if args.header:
        print(f"timestamp{args.delimiter}position_cm")

    # Extract and print data
    tracking_data = data.get("tracking_data", [])
    if not tracking_data:
        print("Warning: No tracking data found", file=sys.stderr)
        sys.exit(0)

    # Check if calibration exists
    metadata = data.get("metadata", {})
    if not metadata.get("calibrated", False):
        print(
            "Warning: Data is not calibrated. Values will be in pixels, not cm.",
            file=sys.stderr,
        )

    for data_point in tracking_data:
        timestamp = data_point.get("timestamp")
        position = extract_position(data_point, args.reference, args.axis, args.offset)

        if timestamp is not None and position is not None:
            print(f"{timestamp}{args.delimiter}{position}")
        elif timestamp is not None:
            # Position is None - print warning or skip
            print(f"{timestamp}{args.delimiter}", file=sys.stderr)


if __name__ == "__main__":
    main()
