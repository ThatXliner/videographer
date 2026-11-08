const half = (n) => Math.floor(n / 2);

// Mirrors ReferencePointSelector.REFERENCE_POINTS; order matches keys 1-9.
export const REFERENCE_POINTS = [
  ["top-left", (x, y, w, h) => [x, y]],
  ["top-center", (x, y, w, h) => [x + half(w), y]],
  ["top-right", (x, y, w, h) => [x + w, y]],
  ["center-left", (x, y, w, h) => [x, y + half(h)]],
  ["center", (x, y, w, h) => [x + half(w), y + half(h)]],
  ["center-right", (x, y, w, h) => [x + w, y + half(h)]],
  ["bottom-left", (x, y, w, h) => [x, y + h]],
  ["bottom-center", (x, y, w, h) => [x + half(w), y + h]],
  ["bottom-right", (x, y, w, h) => [x + w, y + h]],
].map(([name, at]) => ({ name, at }));

export const DEFAULT_REFERENCE_POINT = "bottom-center";

export function referencePoint(name, { x, y, w, h }) {
  return REFERENCE_POINTS.find((p) => p.name === name).at(x, y, w, h);
}
