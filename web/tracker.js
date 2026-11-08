let worker = null;
let engine = null;
// The worker replies in order, so each reply settles the oldest pending request.
let pending = [];

function request(message, transfer = []) {
  return new Promise((resolve, reject) => {
    pending.push({ resolve, reject });
    worker.postMessage(message, transfer);
  });
}

function failAll(error) {
  for (const { reject } of pending) reject(error);
  pending = [];
}

// Starts the worker, which downloads and compiles OpenCV.js off the main thread.
export function loadEngine() {
  engine ??= new Promise((resolve, reject) => {
    worker = new Worker(new URL("./tracker-worker.js", import.meta.url));
    worker.onmessage = ({ data }) => {
      const { resolve, reject } = pending.shift();
      if (data.error) reject(new Error(data.error));
      else resolve(data);
    };
    worker.onerror = (event) => failAll(new Error(event.message || "The tracking engine crashed."));
    request({ type: "load" }).then(resolve, reject);
  }).catch((error) => {
    worker.terminate();
    worker = engine = null;
    throw error;
  });
  return engine;
}

// Adapter over OpenCV's MIL tracker. Frames are full-resolution RGBA ImageData and boxes
// are {x, y, w, h} in video pixels. Each frame's pixels are transferred to the worker, so
// the ImageData is unusable once passed in.
export async function createTracker(frame, bbox) {
  await loadEngine();
  const { width, height, data } = frame;
  await request({ type: "init", width, height, buffer: data.buffer, bbox }, [data.buffer]);
  return {
    update: (next) => request({ type: "update", buffer: next.data.buffer }, [next.data.buffer]),
    dispose: () => request({ type: "dispose" }),
  };
}
