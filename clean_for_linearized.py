#!/usr/bin/env python3
"""
Trim CSV data from start and end until the remaining data is fairly linear.

This utility is designed to extract the linear portion of time-series data,
removing non-linear regions at the beginning and end. Useful for physics
experiments where you want to analyze constant-velocity motion.

The script will:
- Calculate linear regression and R² value for data
- Iteratively trim from start and end to maximize linearity
- Output only the trimmed linear region
- Report statistics about trimming and fit quality
"""

import csv
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import statistics


def calculate_r_squared(data: List[Tuple[float, float]]) -> float:
    """Calculate R² (coefficient of determination) for linear fit.

    Args:
        data: List of (x, y) tuples

    Returns:
        R² value between 0 and 1 (1 = perfect linear fit)
    """
    if len(data) < 2:
        return 0.0

    # Extract x and y values
    x_vals = [x for x, _ in data]
    y_vals = [y for _, y in data]

    # Calculate means
    x_mean = statistics.mean(x_vals)
    y_mean = statistics.mean(y_vals)

    # Calculate slope and intercept using least squares
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
    denominator = sum((x - x_mean) ** 2 for x in x_vals)

    if denominator == 0:
        return 0.0

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    # Calculate R²
    # SS_tot = total sum of squares
    ss_tot = sum((y - y_mean) ** 2 for y in y_vals)

    if ss_tot == 0:
        # All y values are the same
        return 1.0 if all(abs(y - (slope * x + intercept)) < 1e-10 for x, y in zip(x_vals, y_vals)) else 0.0

    # SS_res = residual sum of squares
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_vals, y_vals))

    r_squared = 1 - (ss_res / ss_tot)

    return max(0.0, min(1.0, r_squared))  # Clamp to [0, 1]


def find_linear_region(
    data: List[Tuple[float, float]],
    min_r_squared: float = 0.8,
    min_points: int = 3
) -> Tuple[List[Tuple[float, float]], int, int, float]:
    """Find the longest linear region by trimming from start and end.

    Strategy: Try different combinations of trimming from start and end,
    finding the subset with the highest R² that meets the threshold.

    Args:
        data: List of (timestamp, position) tuples
        min_r_squared: Minimum R² threshold for linearity
        min_points: Minimum number of points to keep

    Returns:
        Tuple of (trimmed_data, points_trimmed_start, points_trimmed_end, final_r_squared)
    """
    n = len(data)

    if n < min_points:
        return data, 0, 0, calculate_r_squared(data)

    # Check if already linear
    current_r2 = calculate_r_squared(data)
    if current_r2 >= min_r_squared:
        return data, 0, 0, current_r2

    # Try trimming combinations
    best_data = data
    best_trim_start = 0
    best_trim_end = 0
    best_r2 = current_r2
    best_length = n

    # Try different trim amounts from start and end
    for trim_start in range(n - min_points + 1):
        for trim_end in range(n - min_points - trim_start + 1):
            if trim_start + trim_end >= n - min_points:
                continue

            # Get trimmed subset
            subset = data[trim_start : n - trim_end] if trim_end > 0 else data[trim_start:]

            if len(subset) < min_points:
                continue

            # Calculate R² for this subset
            r2 = calculate_r_squared(subset)

            # Prefer subsets that meet threshold with maximum length
            # If both meet threshold, prefer longer subset
            # If neither meets threshold, prefer higher R²
            is_better = False

            if r2 >= min_r_squared and best_r2 >= min_r_squared:
                # Both meet threshold - prefer longer
                is_better = len(subset) > best_length
            elif r2 >= min_r_squared and best_r2 < min_r_squared:
                # Only current meets threshold
                is_better = True
            elif r2 < min_r_squared and best_r2 < min_r_squared:
                # Neither meets threshold - prefer higher R² (or longer if equal R²)
                is_better = (r2 > best_r2) or (r2 == best_r2 and len(subset) > best_length)

            if is_better:
                best_data = subset
                best_trim_start = trim_start
                best_trim_end = trim_end
                best_r2 = r2
                best_length = len(subset)

    return best_data, best_trim_start, best_trim_end, best_r2


def main():
    parser = argparse.ArgumentParser(
        description='Trim CSV data to extract the most linear region',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage - extract linear region with R² >= 0.8
  python clean_for_linearized.py data.csv

  # Save to file
  python clean_for_linearized.py data.csv -o linear_data.csv --header

  # More strict linearity requirement
  python clean_for_linearized.py data.csv --min-r2 0.95

  # More lenient linearity requirement
  python clean_for_linearized.py data.csv --min-r2 0.7

  # Require minimum number of points
  python clean_for_linearized.py data.csv --min-points 5

Input format:
  CSV file with two columns: timestamp,position
  - First row can be header (auto-detected)
  - Data should contain some linear region

Output format:
  CSV with trimmed data showing only the linear region
  - Statistics printed to stderr show trim amounts and R² value
  - Sorted by timestamp

Use case:
  Extract constant-velocity portion of motion data, removing acceleration
  and deceleration phases at the start and end.
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
        '--min-r2',
        type=float,
        default=0.8,
        help='Minimum R² threshold for linearity (default: 0.8)'
    )

    parser.add_argument(
        '--min-points',
        type=int,
        default=3,
        help='Minimum number of points to keep (default: 3)'
    )

    parser.add_argument(
        '--header',
        action='store_true',
        help='Include header row in output'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.min_r2 < 0 or args.min_r2 > 1:
        print("Error: --min-r2 must be between 0 and 1", file=sys.stderr)
        sys.exit(1)

    if args.min_points < 2:
        print("Error: --min-points must be at least 2", file=sys.stderr)
        sys.exit(1)

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

    # Calculate initial R²
    initial_r2 = calculate_r_squared(data)

    # Find linear region
    linear_data, trim_start, trim_end, final_r2 = find_linear_region(
        data,
        min_r_squared=args.min_r2,
        min_points=args.min_points
    )

    # Report statistics
    print(f"# Original data points: {len(data)}", file=sys.stderr)
    print(f"# Initial R²: {initial_r2:.6f}", file=sys.stderr)
    print(f"# Points trimmed from start: {trim_start}", file=sys.stderr)
    print(f"# Points trimmed from end: {trim_end}", file=sys.stderr)
    print(f"# Remaining data points: {len(linear_data)}", file=sys.stderr)
    print(f"# Final R²: {final_r2:.6f}", file=sys.stderr)

    if final_r2 < args.min_r2:
        print(f"# WARNING: Could not achieve R² >= {args.min_r2} (best: {final_r2:.6f})", file=sys.stderr)

    # Write output
    output_file = sys.stdout if args.output is None else open(args.output, 'w')

    try:
        writer = csv.writer(output_file, delimiter=args.delimiter)

        if args.header:
            writer.writerow(['timestamp', 'position'])

        for timestamp, position in linear_data:
            writer.writerow([timestamp, position])

    finally:
        if args.output is not None:
            output_file.close()
            print(f"# Linear data written to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
