#!/usr/bin/env python3
"""
Clean CSV data by removing outliers and averaging multiple values per timestamp.

This utility is designed to clean data captured with use-timer, which may have:
1. Outliers from misinterpreted decimal places
2. Multiple data points for the same timestamp

The script will:
- Group data by timestamp
- Remove outliers using IQR (Interquartile Range) method
- Average remaining values for each timestamp
- Output cleaned CSV data
"""

import csv
import argparse
import sys
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
import statistics


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


def clean_data(data: List[Tuple[float, float]], iqr_factor: float = 1.5) -> List[Tuple[float, float]]:
    """Clean data by grouping by timestamp, removing outliers, and averaging.

    Args:
        data: List of (timestamp, position) tuples
        iqr_factor: IQR multiplier for outlier detection

    Returns:
        List of cleaned (timestamp, average_position) tuples
    """
    # Group by timestamp
    timestamp_groups = defaultdict(list)
    for timestamp, position in data:
        timestamp_groups[timestamp].append(position)

    cleaned = []

    for timestamp in sorted(timestamp_groups.keys()):
        positions = timestamp_groups[timestamp]

        if len(positions) == 1:
            # Single value, no cleaning needed
            cleaned.append((timestamp, positions[0]))
        else:
            # Multiple values - detect and remove outliers
            outlier_mask = detect_outliers_iqr(positions, factor=iqr_factor)
            clean_positions = [p for p, is_outlier in zip(positions, outlier_mask) if not is_outlier]

            if len(clean_positions) == 0:
                # All values were outliers - keep the median
                clean_positions = [statistics.median(positions)]

            # Average the remaining values
            avg_position = statistics.mean(clean_positions)
            cleaned.append((timestamp, avg_position))

    return cleaned


def main():
    parser = argparse.ArgumentParser(
        description='Clean CSV data by removing outliers and averaging multiple values per timestamp',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
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
  - Outliers removed using IQR method
  - Multiple values per timestamp averaged
  - Sorted by timestamp
        '''
    )

    parser.add_argument(
        'input_file',
        type=Path,
        help='Path to CSV file with timestamp,position data'
    )

    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        help='Output file path (default: stdout)'
    )

    parser.add_argument(
        '--delimiter',
        default=',',
        help='CSV delimiter (default: comma)'
    )

    parser.add_argument(
        '--iqr-factor',
        type=float,
        default=1.5,
        help='IQR multiplier for outlier detection. Lower = more aggressive (default: 1.5)'
    )

    parser.add_argument(
        '--header',
        action='store_true',
        help='Include header row in output'
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input_file.exists():
        print(f"Error: File not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    # Read CSV data
    data = []
    try:
        with open(args.input_file, 'r') as f:
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
                        print(f"Warning: Skipping row {row_num} - insufficient columns", file=sys.stderr)
                        continue

                    timestamp = float(row[0])
                    position = float(row[1])
                    data.append((timestamp, position))
                except ValueError as e:
                    print(f"Warning: Skipping row {row_num} - invalid data: {e}", file=sys.stderr)
                    continue

    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    if not data:
        print("Error: No valid data found in CSV file", file=sys.stderr)
        sys.exit(1)

    # Clean the data
    cleaned_data = clean_data(data, iqr_factor=args.iqr_factor)

    # Report statistics
    original_count = len(data)
    cleaned_count = len(cleaned_data)
    removed_count = original_count - sum(len(positions) for _, positions in
                                         [(t, [p for ts, p in data if ts == t])
                                          for t, _ in cleaned_data])

    print(f"# Original data points: {original_count}", file=sys.stderr)
    print(f"# Unique timestamps: {cleaned_count}", file=sys.stderr)
    print(f"# Average points per timestamp: {original_count / cleaned_count:.2f}", file=sys.stderr)

    # Write output
    output_file = sys.stdout if args.output is None else open(args.output, 'w')

    try:
        writer = csv.writer(output_file, delimiter=args.delimiter)

        if args.header:
            writer.writerow(['timestamp', 'position'])

        for timestamp, position in cleaned_data:
            writer.writerow([timestamp, position])

    finally:
        if args.output is not None:
            output_file.close()
            print(f"# Cleaned data written to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
