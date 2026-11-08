// Runs OpenCV.js MIL tracking off the main thread. Every message gets exactly one reply, in
// order: {} for load/init/dispose, {success, bbox} for update, or {error}.
const OPENCV_URL = "https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.10.0-release.1/dist/opencv.js";

// The global `cv` may be a Promise, a thenable Emscripten Module, or a plain Module.
// The Module's then() hands back the Module itself, which is still thenable, so awaiting
// it, or resolving any promise with it, spins forever. The engine promise therefore
// resolves to a wrapper object.
function resolveCv(cv) {
  return new Promise((resolve) => {
    if (typeof cv.then === "function") cv.then((module) => resolve({ cv: module }));
    else if (cv.Mat) resolve({ cv });
    else cv.onRuntimeInitialized = () => resolve({ cv });
  });
}

let engine;
try {
  importScripts(OPENCV_URL);
  engine = resolveCv(self.cv);
} catch {
  engine = Promise.reject(new Error("Couldn't download the tracking engine (OpenCV.js)."));
}

let tracker = null;
let frame = null;

// MIL computes its features from channel 0 only, and the CLI feeds it BGR frames, so it
// tracks on blue. Copying just that channel matches the CLI and skips a color conversion.
function blueFrame(buffer) {
  const rgba = new Uint8Array(buffer);
  const blue = frame.data;
  for (let i = 0, j = 2; i < blue.length; i++, j += 4) blue[i] = rgba[j];
  return frame;
}

function dispose() {
  tracker?.delete();
  frame?.delete();
  tracker = frame = null;
}

const handlers = {
  load() {},
  init(cv, { width, height, buffer, bbox }) {
    tracker?.delete();
    if (frame?.cols !== width || frame?.rows !== height) {
      frame?.delete();
      frame = new cv.Mat(height, width, cv.CV_8UC1);
    }
    tracker = new cv.TrackerMIL();
    tracker.init(blueFrame(buffer), new cv.Rect(bbox.x, bbox.y, bbox.w, bbox.h));
  },
  update(cv, { buffer }) {
    const [success, rect] = tracker.update(blueFrame(buffer));
    return {
      success: Boolean(success),
      bbox: {
        x: Math.round(rect.x),
        y: Math.round(rect.y),
        w: Math.round(rect.width),
        h: Math.round(rect.height),
      },
    };
  },
  dispose,
};

self.onmessage = async ({ data }) => {
  try {
    const { cv } = await engine;
    self.postMessage(handlers[data.type](cv, data) ?? {});
  } catch (error) {
    // OpenCV.js throws bare numbers for C++ exceptions.
    self.postMessage({ error: error?.message ?? `OpenCV error ${error}` });
  }
};
