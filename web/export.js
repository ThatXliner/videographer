import { pixelToCm } from "./calibration.js";
import { referencePoint } from "./reference-points.js";

export const CALIBRATION_METHOD = "piecewise linear interpolation with lens distortion correction";

// Python's round() rounds the exact binary value half-to-even; toFixed breaks exact ties upward.
export function pyRound(value, digits) {
  const fixed = value.toFixed(digits);
  const exact = Math.abs(value).toFixed(100);
  const rest = exact.slice(exact.indexOf(".") + 1 + digits);
  if (!/^50*$/.test(rest)) return Number(fixed);
  const scaled = Math.trunc(Math.abs(value) * 10 ** digits);
  const down = scaled % 2 === 0 ? scaled : scaled + 1;
  return (Math.sign(value) * down) / 10 ** digits;
}

const round3 = (v) => pyRound(v, 3);

export function buildEntry({ frame, fps, bbox, referencePointName, ticks }) {
  const { x, y, w, h } = bbox;
  const [refX, refY] = referencePoint(referencePointName, bbox);
  const entry = {
    frame,
    timestamp: round3(fps > 0 ? frame / fps : 0),
    reference_point: referencePointName,
    position_x_pixels: refX,
    position_y_pixels: refY,
    bbox_pixels: { x, y, w, h },
  };
  if (ticks) {
    const [posX, posY] = pixelToCm(ticks, refX, refY);
    const [tlX, tlY] = pixelToCm(ticks, x, y);
    const [brX, brY] = pixelToCm(ticks, x + w, y + h);
    entry.position_x_cm = round3(posX);
    entry.position_y_cm = round3(posY);
    entry.bbox_cm = { x: round3(tlX), y: round3(tlY), w: round3(brX - tlX), h: round3(brY - tlY) };
  }
  return entry;
}

export function buildPositionData({ videoName, referencePointName, entries, ticks, stickLengthCm }) {
  const data = {
    metadata: {
      video_path: videoName,
      output_path: null,
      total_frames: entries.length,
      calibrated: Boolean(ticks),
      reference_point: referencePointName,
    },
    tracking_data: entries,
  };
  if (ticks) {
    data.metadata.calibration = {
      reference_length_cm: stickLengthCm,
      tick_positions: ticks,
      method: CALIBRATION_METHOD,
    };
  }
  return data;
}

export function toCsv({ metadata, tracking_data }) {
  const unit = metadata.calibrated ? "cm" : "pixels";
  const columns = ["timestamp", `position_x_${unit}`, `position_y_${unit}`];
  const rows = tracking_data.map((e) => columns.map((c) => e[c]).join(","));
  return [columns.join(","), ...rows].join("\n") + "\n";
}
