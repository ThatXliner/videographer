import { initialTicks } from "./calibration.js";
import { buildEntry, buildPositionData, toCsv } from "./export.js";
import { plotSvg } from "./plot.js";
import { DEFAULT_REFERENCE_POINT, REFERENCE_POINTS, referencePoint } from "./reference-points.js";
import { createTracker, loadEngine } from "./tracker.js";
import { openVideo } from "./video.js";

const HIT_RADIUS_CSS = 20;
const MIN_BOX = 5;
const COLORS = {
  stick: "#22c55e",
  tick: "#facc15",
  active: "#e879f9",
  box: "#22c55e",
  point: "#ef4444",
  lost: "#ef4444",
};

const $ = (id) => document.getElementById(id);
const stage = $("stage");
const ctx = stage.getContext("2d");
const loupe = $("loupe");
const frameCanvas = new OffscreenCanvas(1, 1);
const frameCtx = frameCanvas.getContext("2d");

const state = {
  phase: "load",
  video: null,
  fps: 30,
  stickLength: 100,
  endpoints: [],
  ticks: null,
  box: null,
  referencePointName: DEFAULT_REFERENCE_POINT,
  pointer: null,
  drag: null,
  run: null,
};

function freshRun() {
  return { entries: [], boxes: [], processed: 0, total: 0, cancelled: false, shown: 0 };
}

function showFrame(image) {
  if (frameCanvas.width !== image.width || frameCanvas.height !== image.height) {
    frameCanvas.width = image.width;
    frameCanvas.height = image.height;
  }
  frameCtx.putImageData(image, 0, 0);
}

let firstFrame = null;
async function showFirstFrame() {
  firstFrame ??= await state.video.frameAt(0, state.fps);
  showFrame(firstFrame);
}

// Video pixels per CSS pixel, so overlays and hit radii stay a constant on-screen size.
function unit() {
  return stage.width / stage.getBoundingClientRect().width;
}

function toVideo(event) {
  const rect = stage.getBoundingClientRect();
  const x = ((event.clientX - rect.left) * stage.width) / rect.width;
  const y = ((event.clientY - rect.top) * stage.height) / rect.height;
  return {
    x: Math.round(Math.max(0, Math.min(stage.width - 1, x))),
    y: Math.round(Math.max(0, Math.min(stage.height - 1, y))),
  };
}

function nearest(points, p) {
  const radius = HIT_RADIUS_CSS * unit();
  let best = null;
  points.forEach(([x, y], index) => {
    const d = Math.hypot(x - p.x, y - p.y);
    if (d <= radius && (!best || d < best.d)) best = { index, d };
  });
  return best?.index ?? null;
}

function circle(x, y, r, fill) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
}

function label(text, x, y, color) {
  const u = unit();
  ctx.font = `600 ${13 * u}px system-ui, sans-serif`;
  ctx.lineWidth = 3 * u;
  ctx.strokeStyle = "rgba(0,0,0,0.75)";
  ctx.strokeText(text, x, y);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

function polyline(points, color, width) {
  ctx.beginPath();
  points.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.strokeStyle = color;
  ctx.lineWidth = width * unit();
  ctx.stroke();
}

function drawBox({ x, y, w, h }, color = COLORS.box) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2 * unit();
  ctx.strokeRect(x, y, w, h);
}

function drawReferencePoint(box) {
  const u = unit();
  const [x, y] = referencePoint(state.referencePointName, box);
  circle(x, y, 7 * u, COLORS.point);
  circle(x, y, 2.5 * u, "#fff");
}

function drawTicks(activeIndex) {
  const u = unit();
  polyline(state.ticks, COLORS.stick, 1.5);
  state.ticks.forEach(([x, y, cm], i) => {
    const color = i === activeIndex ? COLORS.active : COLORS.tick;
    circle(x, y, 6 * u, color);
    if (i % 2 === 0) label(String(cm), x + 8 * u, y - 10 * u, color);
  });
}

function rectFrom(a, b) {
  return {
    x: Math.min(a.x, b.x),
    y: Math.min(a.y, b.y),
    w: Math.abs(a.x - b.x),
    h: Math.abs(a.y - b.y),
  };
}

const phases = {
  load: {
    enter() {
      state.video?.close();
      Object.assign(state, { video: null, endpoints: [], ticks: null, box: null, run: null });
      firstFrame = null;
      $("video-info").hidden = true;
      $("file-input").value = "";
    },
  },

  calibrateEndpoints: {
    primary: "Adjust ticks",
    canConfirm: () => state.endpoints.length === 2 && state.stickLength >= 10,
    confirm() {
      const [a, b] = state.endpoints;
      state.ticks = initialTicks(a, b, state.stickLength);
      go("calibrateTicks");
    },
    back: () => go("load"),
    reset() {
      state.endpoints = [];
    },
    pointerDown(p) {
      const hit = nearest(state.endpoints, p);
      if (hit !== null) state.drag = { endpoint: hit };
      else if (state.endpoints.length < 2) {
        state.endpoints.push([p.x, p.y]);
        state.drag = { endpoint: state.endpoints.length - 1 };
      }
    },
    pointerMove(p) {
      if (state.drag) state.endpoints[state.drag.endpoint] = [p.x, p.y];
    },
    render() {
      const u = unit();
      const [a, b] = state.endpoints;
      const hover = state.pointer && !state.drag && state.endpoints.length === 1 ? state.pointer : null;
      if (a && (b || hover)) {
        const end = b ?? [hover.x, hover.y];
        polyline([a, end], COLORS.stick, 2);
        const length = Math.hypot(end[0] - a[0], end[1] - a[1]);
        label(`${length.toFixed(1)} px`, (a[0] + end[0]) / 2 + 10 * u, (a[1] + end[1]) / 2, COLORS.stick);
      }
      state.endpoints.forEach(([x, y], i) =>
        circle(x, y, 6 * u, state.drag?.endpoint === i ? COLORS.active : COLORS.stick),
      );
      const n = state.endpoints.length;
      $("endpoints-status").textContent =
        n === 0 ? "Mark the 0 cm end first." : n === 1 ? "Now mark the other end." : "Both ends marked.";
    },
  },

  calibrateTicks: {
    primary: "Continue",
    canConfirm: () => true,
    confirm: () => go("selectObject"),
    back: () => go("calibrateEndpoints"),
    reset() {
      const [a, b] = state.endpoints;
      state.ticks = initialTicks(a, b, state.stickLength);
    },
    pointerDown(p) {
      const hit = nearest(state.ticks, p);
      if (hit !== null) state.drag = { tick: hit };
    },
    pointerMove(p) {
      if (!state.drag) return;
      const tick = state.ticks[state.drag.tick];
      tick[0] = p.x;
      tick[1] = p.y;
    },
    render() {
      drawTicks(state.drag?.tick);
    },
  },

  selectObject: {
    primary: "Continue",
    canConfirm: () => state.box !== null,
    confirm: () => go("selectReference"),
    back: () => go(state.ticks ? "calibrateTicks" : "load"),
    reset() {
      state.box = null;
    },
    pointerDown(p) {
      state.drag = { from: p, to: p };
    },
    pointerMove(p) {
      if (state.drag) state.drag.to = p;
    },
    pointerUp() {
      const box = rectFrom(state.drag.from, state.drag.to);
      if (box.w > MIN_BOX && box.h > MIN_BOX) state.box = box;
    },
    render() {
      if (state.ticks) drawTicks();
      if (state.drag) drawBox(rectFrom(state.drag.from, state.drag.to), COLORS.active);
      else if (state.box) drawBox(state.box);
      $("object-status").textContent = state.box
        ? `Box ${state.box.w} × ${state.box.h} px. Drag again to redraw.`
        : "No box yet.";
    },
  },

  selectReference: {
    primary: "Start tracking",
    canConfirm: () => true,
    confirm: () => go("tracking"),
    back: () => go("selectObject"),
    reset() {
      state.referencePointName = DEFAULT_REFERENCE_POINT;
    },
    pointerDown(p) {
      const points = REFERENCE_POINTS.map(({ at }) => at(state.box.x, state.box.y, state.box.w, state.box.h));
      const hit = nearest(points, p);
      if (hit !== null) selectReferencePoint(REFERENCE_POINTS[hit].name);
    },
    key(event) {
      const index = Number(event.key) - 1;
      if (!(index >= 0 && index < REFERENCE_POINTS.length)) return false;
      selectReferencePoint(REFERENCE_POINTS[index].name);
      return true;
    },
    render() {
      const u = unit();
      drawBox(state.box);
      REFERENCE_POINTS.forEach(({ name, at }, i) => {
        const [x, y] = at(state.box.x, state.box.y, state.box.w, state.box.h);
        const selected = name === state.referencePointName;
        circle(x, y, (selected ? 8 : 6) * u, selected ? COLORS.point : "rgba(255,255,255,0.9)");
        label(String(i + 1), x + 9 * u, y - 9 * u, "#fff");
      });
    },
  },

  tracking: {
    get primary() {
      return state.run?.error ? "Back" : "Cancel";
    },
    canConfirm: () => true,
    confirm() {
      if (state.run.error) go("selectReference");
      else state.run.cancelled = true;
    },
    back() {
      phases.tracking.confirm();
    },
    enter() {
      state.run = freshRun();
      track(state.run);
    },
    render() {
      const { boxes, shown } = state.run;
      const frame = boxes[shown];
      if (frame?.success) {
        drawBox(frame.bbox);
        drawReferencePoint(frame.bbox);
      } else if (frame) {
        label("Tracking lost", 12 * unit(), 24 * unit(), COLORS.lost);
      }
    },
  },

  results: {
    primary: "Download JSON",
    canConfirm: () => true,
    confirm: () => download("position_data.json", JSON.stringify(positionData(), null, 2), "application/json"),
    back: () => go("selectReference"),
    enter() {
      const { entries, processed } = state.run;
      const slider = $("frame-slider");
      slider.max = String(processed);
      slider.value = "0";
      state.run.shown = 0;
      const unitName = state.ticks ? "cm" : "pixels";
      $("results-summary").textContent =
        `${entries.length} of ${processed} tracked frames recorded in ${unitName}` +
        (state.run.cancelled ? " (cancelled early)." : ".");
      updateResults();
    },
    render() {
      phases.tracking.render();
    },
  },
};

const STEPS = Object.keys(phases);

function go(phase) {
  state.phase = phase;
  state.drag = null;
  state.pointer = null;
  phases[phase].enter?.();
  syncChrome();
  if (phase === "load") return;
  if (phase !== "tracking" && phase !== "results") showFirstFrame().then(render);
  else render();
}

function syncChrome() {
  const { phase } = state;
  const def = phases[phase];
  document.querySelectorAll("[data-panel]").forEach((el) => {
    el.hidden = el.dataset.panel !== phase;
  });
  $("workspace").hidden = phase === "load";
  $("scrubber").hidden = phase !== "results";
  stage.classList.toggle("interactive", Boolean(def.pointerDown));

  const current = STEPS.indexOf(phase);
  document.querySelectorAll("#steps li").forEach((li) => {
    const index = STEPS.indexOf(li.dataset.step);
    const skipped = !state.ticks && index > 0 && index < 3 && current >= 3;
    li.classList.toggle("done", index < current && !skipped);
    li.classList.toggle("skipped", skipped);
    if (index === current) {
      li.setAttribute("aria-current", "step");
      li.scrollIntoView({ block: "nearest", inline: "nearest" });
    } else li.removeAttribute("aria-current");
  });

  $("primary").textContent = def.primary ?? "";
  $("primary").disabled = !def.canConfirm?.();
  $("reset").hidden = !def.reset;
  $("back").textContent = phase === "tracking" ? "Cancel" : "Back";
  $("back").hidden = phase === "tracking";
  if (phase === "selectReference") renderReferenceGrid();
}

function render() {
  if (state.phase === "load" || frameCanvas.width === 1) return;
  if (stage.width !== frameCanvas.width || stage.height !== frameCanvas.height) {
    stage.width = frameCanvas.width;
    stage.height = frameCanvas.height;
  }
  ctx.drawImage(frameCanvas, 0, 0);
  phases[state.phase].render?.();
  $("primary").disabled = !phases[state.phase].canConfirm?.();
  renderLoupe();
}

function renderLoupe() {
  const p = state.drag && (state.drag.to ?? state.pointer);
  if (!p) {
    loupe.hidden = true;
    return;
  }
  const size = loupe.width;
  const zoom = 3;
  const sourceSize = (size / zoom) * unit() * (loupe.getBoundingClientRect().width / size || 1);
  const lctx = loupe.getContext("2d");
  lctx.imageSmoothingEnabled = false;
  lctx.clearRect(0, 0, size, size);
  lctx.drawImage(stage, p.x - sourceSize / 2, p.y - sourceSize / 2, sourceSize, sourceSize, 0, 0, size, size);
  lctx.strokeStyle = "rgba(255,255,255,0.9)";
  lctx.lineWidth = 1;
  lctx.beginPath();
  lctx.moveTo(size / 2, 0);
  lctx.lineTo(size / 2, size);
  lctx.moveTo(0, size / 2);
  lctx.lineTo(size, size / 2);
  lctx.stroke();

  const rect = stage.getBoundingClientRect();
  const wrap = $("stage-wrap").getBoundingClientRect();
  const sx = rect.left - wrap.left + p.x / unit();
  const sy = rect.top - wrap.top + p.y / unit();
  const box = loupe.getBoundingClientRect().width || 128;
  const gap = 28;
  const left = sx + gap + box > wrap.width ? sx - gap - box : sx + gap;
  const top = sy - gap - box < 0 ? sy + gap : sy - gap - box;
  loupe.style.left = `${Math.max(0, left)}px`;
  loupe.style.top = `${Math.max(0, top)}px`;
  loupe.hidden = false;
}

function selectReferencePoint(name) {
  state.referencePointName = name;
  renderReferenceGrid();
  render();
}

function renderReferenceGrid() {
  const grid = $("ref-grid");
  if (!grid.children.length) {
    REFERENCE_POINTS.forEach(({ name }, i) => {
      const button = document.createElement("button");
      button.type = "button";
      button.role = "radio";
      button.dataset.name = name;
      button.title = name;
      button.innerHTML = `<span class="dot"></span><span class="visually-hidden">${i + 1} ${name}</span>`;
      button.addEventListener("click", () => selectReferencePoint(name));
      grid.append(button);
    });
  }
  for (const button of grid.children) {
    button.setAttribute("aria-checked", String(button.dataset.name === state.referencePointName));
  }
}

async function track(run) {
  $("track-error").hidden = true;
  let tracker;
  try {
    await trackFrames(run, (t) => (tracker = t));
  } catch (e) {
    if (state.run !== run) return;
    run.error = e.message;
    $("track-error").textContent = `Tracking failed: ${e.message}`;
    $("track-error").hidden = false;
    syncChrome();
    return;
  } finally {
    tracker?.dispose?.();
  }
  if (state.run === run) go("results");
}

async function trackFrames(run, keep) {
  const { video, fps, box, referencePointName, ticks } = state;
  run.total = video.frameCount(fps);
  run.boxes[0] = { success: true, bbox: box };
  const first = await video.frameAt(0, fps);
  showFrame(first);
  render();
  $("track-progress").value = 0;
  $("track-status").textContent = "Loading tracking engine…";
  // Decoding runs one frame ahead of the worker, so seeks overlap tracking.
  let next = run.total > 1 ? video.frameAt(1, fps) : null;
  const tracker = keep(await createTracker(first, box));

  let lastPaint = 0;
  for (let i = 1; i < run.total && !run.cancelled; i++) {
    const image = await next;
    if (state.run !== run) return;
    // update() transfers the pixels away, so the preview copy is taken first.
    const now = performance.now();
    const paint = now - lastPaint > 50 || i === run.total - 1;
    if (paint) {
      lastPaint = now;
      showFrame(image);
    }
    const tracked = tracker.update(image);
    next = i + 1 < run.total ? video.frameAt(i + 1, fps) : null;
    const result = await tracked;
    run.boxes[i] = result;
    run.processed = i;
    // Python's track_object labels the first frame after the init frame as frame 0.
    if (result.success) {
      run.entries.push(buildEntry({ frame: i - 1, fps, bbox: result.bbox, referencePointName, ticks }));
    }
    if (paint) {
      run.shown = i;
      render();
      $("track-progress").value = i / (run.total - 1);
      $("track-status").textContent = `Frame ${i} of ${run.total - 1} · ${run.entries.length} tracked`;
    }
  }
}

function positionData() {
  return buildPositionData({
    videoName: state.video.name,
    referencePointName: state.referencePointName,
    entries: state.run.entries,
    ticks: state.ticks,
    stickLengthCm: state.stickLength,
  });
}

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

let scrubToken = 0;
async function updateResults() {
  const run = state.run;
  const index = Number($("frame-slider").value);
  const token = ++scrubToken;
  const image = await state.video.frameAt(index, state.fps);
  if (token !== scrubToken || state.phase !== "results") return;
  run.shown = index;
  showFrame(image);
  render();

  const unitName = state.ticks ? "cm" : "pixels";
  const entry = run.entries.find((e) => e.frame === index - 1);
  const time = (index / state.fps).toFixed(3);
  $("results-readout").textContent =
    index === 0
      ? `Frame 0 (${time} s) · initial selection`
      : entry
        ? `Frame ${index} (${time} s) · x ${entry[`position_x_${unitName}`]}, y ${entry[`position_y_${unitName}`]} ${unitName === "cm" ? "cm" : "px"}`
        : `Frame ${index} (${time} s) · tracking lost`;
  $("plot").innerHTML = plotSvg(run.entries, unitName, entry?.timestamp ?? null);
}

async function loadFile(file) {
  const error = $("load-error");
  error.hidden = true;
  if (!file) return;
  go("load");
  $("dropzone").classList.add("busy");
  try {
    state.video = await openVideo(file);
  } catch (e) {
    error.textContent = e.message;
    error.hidden = false;
    return;
  } finally {
    $("dropzone").classList.remove("busy");
  }
  const { name, width, height, duration, fps } = state.video;
  state.fps = fps;
  $("info-name").textContent = name;
  $("info-size").textContent = `${width} × ${height}`;
  $("info-duration").textContent = `${duration.toFixed(2)} s`;
  $("fps-input").value = String(fps);
  $("video-info").hidden = false;
  $("start-calibration").focus();
}

function onPointer(handlerName) {
  return (event) => {
    const def = phases[state.phase];
    if (!def.pointerDown) return;
    const p = toVideo(event);
    if (handlerName === "pointerDown") {
      if (event.button !== 0) return;
      stage.setPointerCapture(event.pointerId);
      event.preventDefault();
    }
    state.pointer = p;
    if (handlerName === "pointerUp") {
      if (state.drag) def.pointerUp?.(p);
      state.drag = null;
      if (event.pointerType !== "mouse") state.pointer = null;
    } else {
      def[handlerName]?.(p);
    }
    render();
  };
}

stage.addEventListener("pointerdown", onPointer("pointerDown"));
stage.addEventListener("pointermove", onPointer("pointerMove"));
stage.addEventListener("pointerup", onPointer("pointerUp"));
stage.addEventListener("pointercancel", onPointer("pointerUp"));
stage.addEventListener("pointerleave", () => {
  if (!state.drag) {
    state.pointer = null;
    render();
  }
});

$("primary").addEventListener("click", () => phases[state.phase].confirm?.());
$("back").addEventListener("click", () => phases[state.phase].back?.());
$("reset").addEventListener("click", () => {
  phases[state.phase].reset?.();
  render();
});

document.addEventListener("keydown", (event) => {
  if (event.target.closest("input, textarea, select, button, a")) {
    if (event.key !== "Escape" || !event.target.closest("button, a")) return;
  }
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  const def = phases[state.phase];
  if (def.key?.(event)) {
    event.preventDefault();
    return;
  }
  if (event.key === "Enter" && def.canConfirm?.()) def.confirm();
  else if (event.key === "Escape") def.back?.();
  else if (event.key.toLowerCase() === "r" && def.reset) {
    def.reset();
    render();
  } else return;
  event.preventDefault();
});

$("file-input").addEventListener("change", (e) => loadFile(e.target.files[0]));
const dropzone = $("dropzone");
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragging");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragging");
  loadFile(e.dataTransfer.files[0]);
});

$("fps-input").addEventListener("change", (e) => {
  const fps = Number(e.target.value);
  if (fps > 0) state.fps = fps;
  else e.target.value = String(state.fps);
  firstFrame = null;
});
$("stick-length").addEventListener("input", (e) => {
  state.stickLength = Number(e.target.value);
  render();
});
$("start-calibration").addEventListener("click", () => go("calibrateEndpoints"));
$("skip-calibration").addEventListener("click", () => {
  state.endpoints = [];
  state.ticks = null;
  go("selectObject");
});
$("frame-slider").addEventListener("input", updateResults);
$("download-csv").addEventListener("click", () => {
  const base = state.video.name.replace(/\.[^.]+$/, "");
  download(`${base}.csv`, toCsv(positionData()), "text/csv");
});
$("new-video").addEventListener("click", () => go("load"));
window.addEventListener("resize", render);

go("load");
loadEngine().catch(() => {});
