"""The ultimate traditional programmer vs vibe coder test"""

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
from typing import Iterator, List, Tuple


def compress_duplicates(
    data: List[Tuple[float, float]],
) -> Iterator[Tuple[float, float]]:
    # Phase 1: Compress duplicates
    last = data[0]
    for timestamp, position in data:
        if timestamp == last[0]:
            last = (timestamp, (last[1] + position) / 2)
        else:
            yield last
            last = (timestamp, position)
    yield last


def clean_data(
    data: List[Tuple[float, float]], iqr_factor: float = 1.5
) -> Tuple[List[Tuple[float, float]], int]:
    """Clean data by detecting outlier timestamps and averaging duplicate timestamps.

    Args:
        data: List of (timestamp, position) tuples
        iqr_factor: IQR multiplier for outlier detection on timestamps

    Returns:
        Tuple of (cleaned data, number of outliers removed)
    """
    data = list(compress_duplicates(data))
    new_data = []
    for index, row in enumerate(data):
        if index == 0:
            new_data.append(row)
        elif index == len(data) - 1:
            if row[0] > new_data[-1][0]:
                new_data.append(row)
        else:
            first = new_data[-1][0]
            second = data[index][0]
            third = data[index + 1][0]

            print(first, second, third)

            if not (first <= second <= third):
                print("Ditching")
                continue
            else:
                new_data.append(row)
    return new_data, 0
    # if len(data) <= 3:
    #     # Not enough data to detect outliers - just group by timestamp
    #     timestamp_groups = defaultdict(list)
    #     for timestamp, position in data:
    #         timestamp_groups[timestamp].append(position)

    #     cleaned = [
    #         (t, statistics.mean(positions))
    #         for t, positions in sorted(timestamp_groups.items())
    #     ]
    #     return cleaned, 0

    # # Detect outlier timestamps using IQR
    # all_timestamps = [timestamp for timestamp, _ in data]
    # timestamp_outlier_mask = detect_outliers_iqr(all_timestamps, factor=iqr_factor)

    # # Filter out data points with outlier timestamps
    # cleaned_data = [
    #     (t, p)
    #     for (t, p), is_outlier in zip(data, timestamp_outlier_mask)
    #     if not is_outlier
    # ]
    # outliers_removed = len(data) - len(cleaned_data)

    # # Group by timestamp and average positions
    # timestamp_groups = defaultdict(list)
    # for timestamp, position in cleaned_data:
    #     timestamp_groups[timestamp].append(position)

    # # Average positions for each timestamp
    # result = []
    # for timestamp in sorted(timestamp_groups.keys()):
    #     positions = timestamp_groups[timestamp]
    #     avg_position = statistics.mean(positions)
    #     result.append((timestamp, avg_position))

    # return result, outliers_removed


def main():
    parser = argparse.ArgumentParser(
        description="Clean CSV data by removing outliers and averaging multiple values per timestamp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - read from file, output to stdout
  python clean_csv.py data.csv

  # Save to file
  python clean_csv.py data.csv -o cleaned_data.csv

  # Use more aggressive outlier detection
  python clean_csv.py data.csv --iqr-factor 1.0

  # Use less aggressive outlier detection
  python clean_csv.py data.csv --iqr-factor 2.0

  # Specify custom delimiter
  python clean_csv.py data.tsv --delimiter $'\\t'

Input format:
  CSV file with two columns: timestamp,position
  - First row can be header (auto-detected)
  - Timestamp values will be grouped
  - Position values will be cleaned and averaged per timestamp

Output format:
  CSV with one row per unique timestamp
  - Two-pass cleaning: global outliers, then per-timestamp averaging
  - Global pass removes decimal place errors (e.g., 2.407 vs 0.1)
  - Per-timestamp pass removes outliers within duplicate timestamps
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
        help="IQR multiplier for outlier detection. Lower = more aggressive (default: 1.5)",
    )

    parser.add_argument(
        "--header", action="store_true", help="Include header row in output"
    )

    parser.add_argument(
        "--no-global-outliers",
        action="store_true",
        help="Disable global outlier detection (only detect outliers within timestamp groups)",
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
    cleaned_data, outliers_removed = clean_data(
        data,
        # iqr_factor=args.iqr_factor,
        # global_outlier_detection=not args.no_global_outliers
    )

    # Report statistics
    # original_count = len(data)
    # cleaned_count = len(cleaned_data)

    for time, position in cleaned_data:
        print(f"{time}\t{position}")

    # print(f"# Original data points: {original_count}", file=sys.stderr)
    # print(f"# Outliers removed: {outliers_removed}", file=sys.stderr)
    # print(f"# Unique timestamps: {cleaned_count}", file=sys.stderr)
    # if cleaned_count > 0:
    #     print(
    #         f"# Average points per timestamp: {(original_count - outliers_removed) / cleaned_count:.2f}",
    #         file=sys.stderr,
    #     )

    # # Write output
    # output_file = sys.stdout if args.output is None else open(args.output, "w")

    # try:
    #     writer = csv.writer(output_file, delimiter=args.delimiter)

    #     if args.header:
    #         writer.writerow(["timestamp", "position"])

    #     for timestamp, position in cleaned_data:
    #         writer.writerow([timestamp, position])

    # finally:
    #     if args.output is not None:
    #         output_file.close()
    #         print(f"# Cleaned data written to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
