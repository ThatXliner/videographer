"""Main entry point for VideoGrapher object tracking application."""

import argparse
import sys
from pathlib import Path

from .tracker import ObjectTracker


def main():
    parser = argparse.ArgumentParser(
        prog="videographer",
        description="Track objects in videos with precise position measurements and lens distortion correction",
        epilog="Example: videographer input.mp4 -o tracked.mp4 --no-calibration",
    )

    # Positional arguments
    parser.add_argument("video_path", type=str, help="Path to input video file")

    # Optional arguments
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.mp4",
        help="Path to output video file (default: output.mp4)",
    )

    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip scale calibration step (track in pixels only)",
    )

    parser.add_argument(
        "--reference-length",
        type=float,
        default=100.0,
        metavar="CM",
        help="Length of reference object in centimeters (default: 100.0 for meter stick)",
    )

    parser.add_argument(
        "--output-data",
        type=str,
        default="position_data.json",
        metavar="FILE",
        help="Path to output JSON data file (default: position_data.json)",
    )

    parser.add_argument("-v", "--version", action="version", version="%(prog)s 0.1.0")

    args = parser.parse_args()

    # Validate input file exists
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video_path}", file=sys.stderr)
        sys.exit(1)

    # Create tracker and run
    tracker = ObjectTracker(str(video_path), args.output)
    tracker.run(
        calibrate=not args.no_calibration, reference_length_cm=args.reference_length
    )

    # Save position data with custom filename if specified
    if args.output_data != "position_data.json":
        tracker.save_position_data(args.output_data)


if __name__ == "__main__":
    main()
