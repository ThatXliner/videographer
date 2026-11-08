"""Write the Python reference values the web parity tests compare against.

Run from the repo root: uv run python web/tests/gen_fixtures.py
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from main import MeterStickCalibrator, ObjectTracker, ReferencePointSelector  # noqa: E402

OUT = Path(__file__).parent / "fixtures"


def tick_layouts():
    yield [[100 + 50 * i, 300, 10 * i] for i in range(11)]
    yield [[40 + 31 * i, 20 + 17 * i, 10 * i] for i in range(4)]
    yield [[600 - 45 * i, 100, 10 * i] for i in range(11)]
    yield [[250, 50 + 40 * i, 10 * i] for i in range(6)]
    barrel = []
    for i in range(11):
        u = (i - 5) / 5
        barrel.append([int(320 + 280 * u * (1 - 0.12 * u * u)), int(400 - 30 * (1 - u * u)), 10 * i])
    yield barrel
    yield [[100, 100, 0], [100, 100, 10], [160, 110, 20], [230, 100, 30]]


def pixel_to_cm_cases(rng):
    cases = []
    for ticks in tick_layouts():
        tracker = ObjectTracker.__new__(ObjectTracker)
        tracker.calibration_data = {"tick_positions": ticks}
        points = [(x, y) for x in range(-50, 751, 80) for y in range(-20, 521, 60)]
        points += [(rng.uniform(-100, 800), rng.uniform(-100, 600)) for _ in range(60)]
        points += [(t[0], t[1]) for t in ticks]
        cases.append(
            {
                "ticks": ticks,
                "points": [
                    [px, py, *map(float, tracker._pixel_to_cm(px, py))] for px, py in points
                ],
            }
        )
    return cases


def initial_tick_cases():
    cases = []
    for start, end, length in [
        ((10, 20), (610, 20), 100.0),
        ((603, 411), (17, 95), 100.0),
        ((0, 0), (333, 77), 30.0),
        ((500, 10), (12, 470), 55.0),
    ]:
        calibrator = MeterStickCalibrator()
        calibrator.start_point, calibrator.end_point = start, end
        calibrator.stick_length_cm = length
        calibrator._initialize_tick_positions()
        cases.append({"start": start, "end": end, "length": length, "ticks": calibrator.tick_positions})
    return cases


def reference_point_cases():
    boxes = [(10, 20, 30, 40), (0, 0, 7, 9), (123, 45, 61, 33), (5, 5, 6, 6)]
    return [
        {"name": name, "box": box, "point": list(fn(*box))}
        for name, fn in ReferencePointSelector.REFERENCE_POINTS.values()
        for box in boxes
    ]


def rounding_cases(rng):
    values = [0.0625, 0.0005, 1.0005, 2.675, -0.0625, 12.3455, 0.1235, 1.2345, -7.0015]
    values += [rng.uniform(-500, 500) for _ in range(300)]
    values += [i / 16 for i in range(-80, 80)]
    return [[v, round(v, 3)] for v in values]


def main():
    rng = random.Random(1234)
    OUT.mkdir(exist_ok=True)
    fixtures = {
        "pixel_to_cm.json": pixel_to_cm_cases(rng),
        "initial_ticks.json": initial_tick_cases(),
        "reference_points.json": reference_point_cases(),
        "rounding.json": rounding_cases(rng),
    }
    for name, data in fixtures.items():
        (OUT / name).write_text(json.dumps(data) + "\n")
        print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
