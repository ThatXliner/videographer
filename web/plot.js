const WIDTH = 360;
const HEIGHT = 220;
const PAD = { left: 44, right: 10, top: 10, bottom: 34 };

function niceTicks(min, max, count = 4) {
  if (min === max) return [min];
  const raw = (max - min) / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * magnitude).find((s) => s >= raw);
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) {
    ticks.push(Math.round(v / step) * step);
  }
  return ticks;
}

const label = (v) => String(Number(v.toPrecision(6)));

// Renders x and y position against time as an SVG string. Colors come from CSS classes.
export function plotSvg(entries, unit, cursorTime = null) {
  if (entries.length === 0) return "";
  const xKey = `position_x_${unit}`;
  const yKey = `position_y_${unit}`;
  const times = entries.map((e) => e.timestamp);
  const values = entries.flatMap((e) => [e[xKey], e[yKey]]);
  const t0 = Math.min(...times);
  const t1 = Math.max(...times, t0 + 1e-3);
  let v0 = Math.min(...values);
  let v1 = Math.max(...values);
  if (v0 === v1) [v0, v1] = [v0 - 1, v1 + 1];

  const sx = (t) => PAD.left + ((t - t0) / (t1 - t0)) * (WIDTH - PAD.left - PAD.right);
  const sy = (v) => HEIGHT - PAD.bottom - ((v - v0) / (v1 - v0)) * (HEIGHT - PAD.top - PAD.bottom);
  const line = (key) =>
    entries.map((e) => `${sx(e.timestamp).toFixed(1)},${sy(e[key]).toFixed(1)}`).join(" ");

  const yGrid = niceTicks(v0, v1)
    .map(
      (v) =>
        `<line class="grid" x1="${PAD.left}" x2="${WIDTH - PAD.right}" y1="${sy(v)}" y2="${sy(v)}"/>` +
        `<text class="tick" x="${PAD.left - 6}" y="${sy(v)}" text-anchor="end" dominant-baseline="middle">${label(v)}</text>`,
    )
    .join("");
  const xTicks = niceTicks(t0, t1)
    .map(
      (t) =>
        `<text class="tick" x="${sx(t)}" y="${HEIGHT - PAD.bottom + 16}" text-anchor="middle">${label(t)}</text>`,
    )
    .join("");
  const cursor =
    cursorTime === null || cursorTime < t0 || cursorTime > t1
      ? ""
      : `<line class="cursor" x1="${sx(cursorTime)}" x2="${sx(cursorTime)}" y1="${PAD.top}" y2="${HEIGHT - PAD.bottom}"/>`;

  return `<svg viewBox="0 0 ${WIDTH} ${HEIGHT}" role="img" aria-label="Position in ${unit} against time in seconds">
  ${yGrid}${xTicks}
  <text class="axis-label" x="${(WIDTH + PAD.left) / 2}" y="${HEIGHT - 4}" text-anchor="middle">time (s)</text>
  <text class="axis-label" x="12" y="${(HEIGHT - PAD.bottom) / 2}" text-anchor="middle" transform="rotate(-90 12 ${(HEIGHT - PAD.bottom) / 2})">${unit}</text>
  ${cursor}
  <polyline class="series series-x" points="${line(xKey)}"/>
  <polyline class="series series-y" points="${line(yKey)}"/>
</svg>`;
}
