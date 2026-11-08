const OPENCV_URL = "https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.10.0-release.1/dist/opencv.js";

let engine = null;

// The global `cv` may be a Promise, a thenable Emscripten Module, or a plain Module.
// The Module's then() hands back the Module itself, which is still thenable, so awaiting
// it, or resolving any promise with it, spins forever and freezes the page. The engine
// promise therefore resolves to a wrapper object.
function resolveCv(cv) {
  return new Promise((resolve) => {
    if (typeof cv.then === "function") cv.then((module) => resolve({ cv: module }));
    else if (cv.Mat) resolve({ cv });
    else cv.onRuntimeInitialized = () => resolve({ cv });
  });
}

export function loadEngine() {
  engine ??= new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = OPENCV_URL;
    script.async = true;
    script.onload = () => resolveCv(globalThis.cv).then(resolve, reject);
    script.onerror = () => reject(new Error("Couldn't download the tracking engine (OpenCV.js)."));
    document.head.append(script);
  }).catch((error) => {
    engine = null;
    throw error;
  });
  return engine;
}

function toRgbMat(cv, imageData) {
  const rgba = cv.matFromImageData(imageData);
  const rgb = new cv.Mat();
  cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB);
  rgba.delete();
  return rgb;
}

// Adapter over OpenCV's MIL tracker. Frames are full-resolution RGBA ImageData and boxes
// are {x, y, w, h} in video pixels.
export async function createTracker(frame, bbox) {
  const { cv } = await loadEngine();
  const tracker = new cv.TrackerMIL();
  const first = toRgbMat(cv, frame);
  try {
    tracker.init(first, new cv.Rect(bbox.x, bbox.y, bbox.w, bbox.h));
  } finally {
    first.delete();
  }

  return {
    update(next) {
      const mat = toRgbMat(cv, next);
      try {
        const [success, rect] = tracker.update(mat);
        return {
          success: Boolean(success),
          bbox: {
            x: Math.round(rect.x),
            y: Math.round(rect.y),
            w: Math.round(rect.width),
            h: Math.round(rect.height),
          },
        };
      } finally {
        mat.delete();
      }
    },
    dispose() {
      tracker.delete();
    },
  };
}
