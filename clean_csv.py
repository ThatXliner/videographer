#!/usr/bin/env python3
"""
Clean CSV data by removing outlier timestamps and averaging duplicate timestamps.

This utility is designed to clean data captured with use-timer, which may have:
1. Outlier timestamps from misinterpreted decimal places (e.g., 2.407 instead of 0.1)
2. Multiple data points for the same timestamp

The script will:
- Detect and remove outlier timestamps using IQR (Interquartile Range) method
- Group remaining data by timestamp
- Average position values for duplicate timestamps
- Output cleaned CSV data

Note: Position values are NEVER filtered as outliers - only timestamps are checked.
"""

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple


def detect_outliers_iqr(values: List[float], factor: float = 1.5) -> List[bool]:
    """Detect outliers using the IQR (Interquartile Range) method.

    Args:
        values: List of numeric values
        factor: IQR multiplier for outlier bounds (default: 1.5 for standard outliers)

    Returns:
        List of booleans where True indicates an outlier
    """
    if len(values) <= 3:
        # Not enough data to detect outliers reliably
        return [False] * len(values)

    # Calculate quartiles
    q1 = statistics.quantiles(values, n=4)[0]  # 25th percentile
    q3 = statistics.quantiles(values, n=4)[2]  # 75th percentile
    iqr = q3 - q1

    # Define outlier bounds
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    # Mark outliers
    return [v < lower_bound or v > upper_bound for v in values]


def clean_data(
    data: List[Tuple[float, float]], iqr_factor: float = 1.5
) -> Tuple[List[Tuple[float, float]], int]:
    """Clean data by detecting outlier timestamps and averaging duplicate timestamps.

    Outlier detection includes:
    1. IQR-based detection for timestamps far from the main distribution
    2. Zero timestamps that appear after non-zero timestamps (erroneous resets)

    Args:
        data: List of (timestamp, position) tuples
        iqr_factor: IQR multiplier for outlier detection on timestamps

    Returns:
        Tuple of (cleaned data, number of outliers removed)
    """
    if len(data) <= 3:
        # Not enough data to detect outliers - just group by timestamp
        timestamp_groups = defaultdict(list)
        for timestamp, position in data:
            timestamp_groups[timestamp].append(position)

        cleaned = [
            (t, statistics.mean(positions))
            for t, positions in sorted(timestamp_groups.items())
        ]
        return cleaned, 0

    # Detect outlier timestamps using IQR
    all_timestamps = [timestamp for timestamp, _ in data]
    timestamp_outlier_mask = detect_outliers_iqr(all_timestamps, factor=iqr_factor)

    # Additional check: flag zero timestamps that appear after non-zero timestamps
    seen_nonzero = False
    for i, (timestamp, _) in enumerate(data):
        if timestamp > 0:
            seen_nonzero = True
        elif timestamp == 0 and seen_nonzero:
            # Zero appearing after we've seen non-zero timestamps - mark as outlier
            timestamp_outlier_mask[i] = True

    # Filter out data points with outlier timestamps
    cleaned_data = [
        (t, p)
        for (t, p), is_outlier in zip(data, timestamp_outlier_mask)
        if not is_outlier
    ]
    outliers_removed = len(data) - len(cleaned_data)

    # Group by timestamp and average positions
    timestamp_groups = defaultdict(list)
    for timestamp, position in cleaned_data:
        timestamp_groups[timestamp].append(position)

    # Average positions for each timestamp
    result = []
    for timestamp in sorted(timestamp_groups.keys()):
        positions = timestamp_groups[timestamp]
        avg_position = statistics.mean(positions)
        result.append((timestamp, avg_position))

    return result, outliers_removed


def main():
    parser = argparse.ArgumentParser(
        description="Clean CSV data by removing outliers and averaging multiple values per timestamp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - read from file, output to stdout
  python clean_csv.py data.csv

  # Save to file
  python clean_csv.py data.csv -o cleaned_data.csv --header

  # Use more aggressive outlier detection (lower IQR factor)
  python clean_csv.py data.csv --iqr-factor 1.0

  # Use less aggressive outlier detection (higher IQR factor)
  python clean_csv.py data.csv --iqr-factor 2.0

  # Specify custom delimiter
  python clean_csv.py data.tsv --delimiter $'\\t'

Input format:
  CSV file with two columns: timestamp,position
  - First row can be header (auto-detected)
  - Timestamps may have errors from decimal place misreading

Output format:
  CSV with one row per unique timestamp
  - Outlier timestamps removed using IQR method (e.g., 2.407 when others are ~0.1)
  - Duplicate timestamps averaged
  - Position values are NEVER filtered as outliers
  - Sorted by timestamp
        """,
    )

    parser.add_argument(
        "input_file", type=Path, help="Path to CSV file with timestamp,position data"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: stdout)",
    )

    parser.add_argument(
        "--delimiter", default=",", help="CSV delimiter (default: comma)"
    )

    parser.add_argument(
        "--iqr-factor",
        type=float,
        default=1.5,
        help="IQR multiplier for timestamp outlier detection. Lower = more aggressive (default: 1.5)",
    )

    parser.add_argument(
        "--header", action="store_true", help="Include header row in output"
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input_file.exists():
        print(f"Error: File not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    # Read CSV data
    data = []
    try:
        with open(args.input_file, "r") as f:
            reader = csv.reader(f, delimiter=args.delimiter)

            # Read first row to check if it's a header
            first_row = next(reader, None)
            if first_row is None:
                print("Error: CSV file is empty", file=sys.stderr)
                sys.exit(1)

            # Try to parse first row as data
            try:
                timestamp = float(first_row[0])
                position = float(first_row[1])
                data.append((timestamp, position))
            except (ValueError, IndexError):
                # First row is likely a header, skip it
                pass

            # Read remaining rows
            for row_num, row in enumerate(reader, start=2):
                try:
                    if len(row) < 2:
                        print(
                            f"Warning: Skipping row {row_num} - insufficient columns",
                            file=sys.stderr,
                        )
                        continue

                    timestamp = float(row[0])
                    position = float(row[1])
                    data.append((timestamp, position))
                except ValueError as e:
                    print(
                        f"Warning: Skipping row {row_num} - invalid data: {e}",
                        file=sys.stderr,
                    )
                    continue

    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    if not data:
        print("Error: No valid data found in CSV file", file=sys.stderr)
        sys.exit(1)

    # Clean the data
    cleaned_data, outliers_removed = clean_data(data, iqr_factor=args.iqr_factor)

    # Report statistics
    original_count = len(data)
    cleaned_count = len(cleaned_data)

    print(f"# Original data points: {original_count}", file=sys.stderr)
    print(f"# Outliers removed: {outliers_removed}", file=sys.stderr)
    print(f"# Unique timestamps: {cleaned_count}", file=sys.stderr)
    if cleaned_count > 0:
        print(
            f"# Average points per timestamp: {(original_count - outliers_removed) / cleaned_count:.2f}",
            file=sys.stderr,
        )

    for time, position in cleaned_data:
        print(f"{time},{position}")


if __name__ == "__main__":
    main()
