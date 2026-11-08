export function initialTicks([sx, sy], [ex, ey], stickLengthCm) {
  const count = Math.floor(stickLengthCm / 10) + 1;
  const dx = ex - sx;
  const dy = ey - sy;
  return Array.from({ length: count }, (_, i) => {
    const t = i / (count - 1);
    return [Math.trunc(sx + t * dx), Math.trunc(sy + t * dy), i * 10];
  });
}

// Faithful port of ObjectTracker._pixel_to_cm in main.py, quirks included:
// the closest segment uses a clamped projection, but interpolation along it does not.
export function pixelToCm(ticks, px, py) {
  let minDist = Infinity;
  let closest = 0;
  for (let i = 0; i < ticks.length - 1; i++) {
    const [x1, y1] = ticks[i];
    const [x2, y2] = ticks[i + 1];
    const dx = x2 - x1;
    const dy = y2 - y1;
    if (dx === 0 && dy === 0) continue;
    const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)));
    const ox = px - (x1 + t * dx);
    const oy = py - (y1 + t * dy);
    const dist = Math.sqrt(ox * ox + oy * oy);
    if (dist < minDist) {
      minDist = dist;
      closest = i;
    }
  }

  const [x1, y1, cm1] = ticks[closest];
  const [x2, y2, cm2] = ticks[Math.min(closest + 1, ticks.length - 1)];
  const dx = x2 - x1;
  const dy = y2 - y1;
  const segmentLength = Math.sqrt(dx * dx + dy * dy);
  if (segmentLength === 0) return [cm1, 0];

  const t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy);
  const cmPerPixel = Math.abs(cm2 - cm1) / segmentLength;
  return [cm1 + t * (cm2 - cm1), minDist * cmPerPixel];
}
