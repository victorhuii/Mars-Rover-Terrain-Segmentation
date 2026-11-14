import {
  classify,
  imageKind,
  letterbox,
  MAX_BYTES,
  MAX_PIXELS,
  SIZE,
} from "./terrain";
import { resizeRgb } from "./resize";
import type { Geometry } from "./terrain";
import type { InferenceSession } from "onnxruntime-web";

const MODEL_URL = `${import.meta.env.BASE_URL}models/marsterrain-v1-be2513a68f69.onnx`;
export type SourceImage = {
  image: HTMLImageElement;
  name: string;
  width: number;
  height: number;
  sampleId?: string;
};
export type Prediction = {
  labels: Uint8Array;
  counts: number[];
  total: number;
  region: Geometry;
  elapsedMs: number;
  loadMs: number;
};

type Progress = (message: string) => void;
let sessionPromise: Promise<InferenceSession> | undefined;

export async function decodeFile(file: File): Promise<SourceImage> {
  if (!file.size)
    throw new Error("This file is empty. Choose a JPEG, PNG or WebP image.");
  if (file.size > MAX_BYTES)
    throw new Error("That file is larger than 10 MB. Choose a smaller image.");
  const kind = imageKind(new Uint8Array(await file.slice(0, 12).arrayBuffer()));
  if (!kind)
    throw new Error("That file is not a supported JPEG, PNG or WebP image.");
  const mime: Record<string, string> = {
    jpeg: "image/jpeg",
    png: "image/png",
    webp: "image/webp",
  };
  if (file.type && file.type !== mime[kind])
    throw new Error("The file type does not match its contents.");
  const objectUrl = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.decoding = "async";
    await new Promise<void>((resolve, reject) => {
      image.onload = () => resolve();
      image.onerror = () =>
        reject(new Error("The image could not be decoded. Try another file."));
      image.src = objectUrl;
    });
    const width = image.naturalWidth;
    const height = image.naturalHeight;
    if (
      !width ||
      !height ||
      width * height > MAX_PIXELS ||
      Math.max(width, height) > 8192
    )
      throw new Error(
        "Image dimensions exceed the 20-megapixel or 8,192-pixel side limit.",
      );
    return { image, name: file.name, width, height };
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

export function prepareImage(source: SourceImage) {
  const region = letterbox(source.width, source.height);
  const canvas = document.createElement("canvas");
  canvas.width = source.width;
  canvas.height = source.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Your browser could not prepare this image.");
  context.fillStyle = "#000";
  context.fillRect(0, 0, source.width, source.height);
  context.drawImage(source.image, 0, 0);
  const pixels = context.getImageData(0, 0, source.width, source.height).data;
  const resized = resizeRgb(
    pixels,
    source.width,
    source.height,
    region.width,
    region.height,
  );
  const tensor = new Float32Array(3 * SIZE * SIZE);
  for (let y = 0; y < region.height; y++)
    for (let x = 0; x < region.width; x++) {
      const destination = (y + region.y) * SIZE + x + region.x;
      const sourcePixel = (y * region.width + x) * 3;
      for (let c = 0; c < 3; c++)
        tensor[c * SIZE * SIZE + destination] = resized[sourcePixel + c] / 255;
    }
  return { tensor, region };
}

async function getSession(progress: Progress) {
  const ort = await import("onnxruntime-web/wasm");
  ort.env.wasm.wasmPaths = `${import.meta.env.BASE_URL}ort/`;
  ort.env.wasm.numThreads = 1;
  if (!sessionPromise) {
    sessionPromise = (async () => {
      progress("Downloading the model…");
      const response = await fetch(MODEL_URL);
      if (!response.ok)
        throw new Error(
          "The model could not be downloaded. Check your connection and try again.",
        );
      const length = Number(response.headers.get("content-length") || 0);
      let buffer: ArrayBuffer;
      if (response.body && length > 0) {
        const reader = response.body.getReader();
        const chunks: Uint8Array[] = [];
        let loaded = 0;
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          loaded += value.byteLength;
          progress(
            `Downloading the model… ${Math.min(100, Math.round((loaded / length) * 100))}%`,
          );
        }
        const merged = new Uint8Array(loaded);
        let offset = 0;
        for (const chunk of chunks) {
          merged.set(chunk, offset);
          offset += chunk.byteLength;
        }
        buffer = merged.buffer;
      } else {
        buffer = await response.arrayBuffer();
      }
      progress("Preparing the model in your browser…");
      return ort.InferenceSession.create(buffer, {
        executionProviders: ["wasm"],
      });
    })().catch((error) => {
      sessionPromise = undefined;
      throw error;
    });
  }
  return { ort, session: await sessionPromise };
}

export async function analyze(
  source: SourceImage,
  progress: Progress,
): Promise<Prediction> {
  const loadStart = performance.now();
  const { ort, session } = await getSession(progress);
  const loadMs = performance.now() - loadStart;
  progress("Analyzing terrain…");
  await new Promise((resolve) => setTimeout(resolve, 0));
  const { tensor, region } = prepareImage(source);
  const start = performance.now();
  const output = await session.run({
    image: new ort.Tensor("float32", tensor, [1, 3, SIZE, SIZE]),
  });
  const elapsedMs = performance.now() - start;
  const logits = output.logits;
  if (
    !logits ||
    !(logits.data instanceof Float32Array) ||
    logits.dims.join(",") !== `1,4,${SIZE},${SIZE}`
  )
    throw new Error("The model returned an unexpected output.");
  return { ...classify(logits.data, region), region, elapsedMs, loadMs };
}
