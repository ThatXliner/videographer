import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { initialTicks, pixelToCm } from "../calibration.js";
import { REFERENCE_POINTS, referencePoint } from "../reference-points.js";
import { pyRound } from "../export.js";

const fixture = (name) => JSON.parse(readFileSync(new URL(`fixtures/${name}`, import.meta.url)));

test("pixelToCm matches Python _pixel_to_cm", () => {
  for (const { ticks, points } of fixture("pixel_to_cm.json")) {
    for (const [px, py, xCm, yCm] of points) {
      const [jx, jy] = pixelToCm(ticks, px, py);
      assert.ok(Math.abs(jx - xCm) < 1e-9, `x at (${px}, ${py}): ${jx} vs ${xCm}`);
      assert.ok(Math.abs(jy - yCm) < 1e-9, `y at (${px}, ${py}): ${jy} vs ${yCm}`);
    }
  }
});

test("initialTicks matches Python _initialize_tick_positions", () => {
  for (const { start, end, length, ticks } of fixture("initial_ticks.json")) {
    assert.deepEqual(initialTicks(start, end, length), ticks);
  }
});

test("reference point registry matches Python REFERENCE_POINTS", () => {
  const cases = fixture("reference_points.json");
  assert.deepEqual(
    REFERENCE_POINTS.map((p) => p.name),
    [...new Set(cases.map((c) => c.name))],
  );
  for (const { name, box: [x, y, w, h], point } of cases) {
    assert.deepEqual(referencePoint(name, { x, y, w, h }), point, `${name} on ${[x, y, w, h]}`);
  }
});

test("pyRound matches Python round(v, 3)", () => {
  for (const [value, expected] of fixture("rounding.json")) {
    assert.equal(pyRound(value, 3), expected, `round(${value}, 3)`);
  }
});
