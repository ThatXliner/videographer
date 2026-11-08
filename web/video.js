const COMMON_FPS = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 120];
const FALLBACK_FPS = 30;

function once(target, event, errorEvent = "error") {
  return new Promise((resolve, reject) => {
    const ok = () => {
      target.removeEventListener(errorEvent, fail);
      resolve();
    };
    const fail = () => {
      target.removeEventListener(event, ok);
      reject(target.error ?? new Error(`${errorEvent} while waiting for ${event}`));
    };
    target.addEventListener(event, ok, { once: true });
    target.addEventListener(errorEvent, fail, { once: true });
  });
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

async function detectFps(video) {
  if (!("requestVideoFrameCallback" in HTMLVideoElement.prototype)) return FALLBACK_FPS;
  const times = [];
  const sampled = new Promise((resolve) => {
    const onFrame = (_, { mediaTime }) => {
      times.push(mediaTime);
      if (times.length < 16) video.requestVideoFrameCallback(onFrame);
      else resolve();
    };
    video.requestVideoFrameCallback(onFrame);
  });
  const timeout = new Promise((resolve) => setTimeout(resolve, 1500));
  try {
    await video.play();
    await Promise.race([sampled, timeout]);
  } catch {
    return FALLBACK_FPS;
  } finally {
    video.pause();
  }

  const deltas = times.slice(1).map((t, i) => t - times[i]).filter((d) => d > 0);
  if (deltas.length < 4) return FALLBACK_FPS;
  const fps = 1 / median(deltas);
  const nearest = COMMON_FPS.reduce((a, b) => (Math.abs(b - fps) < Math.abs(a - fps) ? b : a));
  return Math.abs(nearest - fps) / nearest < 0.01 ? nearest : Math.round(fps * 100) / 100;
}

// Owns a detached <video> and reads exact frames by seeking to the middle of each frame.
export async function openVideo(file) {
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  // Kept in the DOM and barely visible: some browsers skip frame callbacks for detached
  // or fully transparent videos.
  video.style.cssText =
    "position:fixed;left:0;bottom:0;width:2px;height:2px;opacity:0.01;pointer-events:none";
  document.body.append(video);
  video.src = url;

  try {
    await once(video, "loadeddata");
  } catch {
    video.remove();
    URL.revokeObjectURL(url);
    throw new Error(
      `This browser can't decode “${file.name}”. Try an MP4 (H.264) or WebM file, or convert it first.`,
    );
  }

  // Recorded WebM files often omit their duration until the end has been seen.
  if (!Number.isFinite(video.duration)) {
    const seeked = once(video, "seeked");
    video.currentTime = Number.MAX_SAFE_INTEGER;
    await seeked;
    video.currentTime = 0;
    await once(video, "seeked");
  }

  const { videoWidth: width, videoHeight: height, duration } = video;
  const fps = await detectFps(video);
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  let queue = Promise.resolve();
  async function readFrame(index, frameRate) {
    const time = (index + 0.5) / frameRate;
    if (Math.abs(video.currentTime - time) > 1e-6) {
      const seeked = once(video, "seeked");
      video.currentTime = time;
      await seeked;
    }
    ctx.drawImage(video, 0, 0, width, height);
    return ctx.getImageData(0, 0, width, height);
  }

  return {
    name: file.name,
    width,
    height,
    duration,
    fps,
    frameCount: (frameRate) => Math.max(1, Math.floor(duration * frameRate)),
    // Seeks are serialized because a <video> shows one position at a time.
    frameAt(index, frameRate) {
      const result = queue.then(() => readFrame(index, frameRate));
      queue = result.catch(() => {});
      return result;
    },
    close() {
      video.removeAttribute("src");
      video.load();
      video.remove();
      URL.revokeObjectURL(url);
    },
  };
}
