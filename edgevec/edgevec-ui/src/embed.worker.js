/**
 * EdgeVec: Client-Side Web Worker Neural Inference Thread
 * Department of Computer Science & Engineering, National Institute of Technology Patna
 *
 * Encapsulates the ONNX Runtime WebAssembly environment in a dedicated Worker.
 * Executes feature extraction (all-MiniLM-L6-v2) using SIMD 128-bit vector extensions
 * without blocking UI thread event loops.
 */

import { pipeline, env } from "@huggingface/transformers";

// Worker-specific config — no need to check for DOM APIs
env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

const MODEL_NAME = "Xenova/all-MiniLM-L6-v2";
let _embedder = null;

async function loadModel(onProgress) {
  if (_embedder) return _embedder;

  _embedder = await pipeline("feature-extraction", MODEL_NAME, {
    progress_callback: (prog) => {
      if (prog.status === "downloading" && prog.total) {
        onProgress(Math.round((prog.loaded / prog.total) * 100), prog.file);
      }
    },
  });

  return _embedder;
}

self.addEventListener("message", async (event) => {
  const { type, payload } = event.data;

  if (type === "embed") {
    try {
      const embedder = await loadModel((pct, file) => {
        self.postMessage({ type: "progress", payload: { pct, file } });
      });

      const output = await embedder(payload.text, {
        pooling: "mean",
        normalize: true,
      });

      self.postMessage({
        type: "result",
        payload: { vector: Array.from(output.data) },
      });
    } catch (err) {
      self.postMessage({
        type: "error",
        payload: { message: err.message },
      });
    }
  }
});
