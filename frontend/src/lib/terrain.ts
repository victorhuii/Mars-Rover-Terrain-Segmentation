export const SIZE = 256;
export const MAX_BYTES = 10 * 1024 * 1024;
export const MAX_PIXELS = 20_000_000;
export const CLASSES = [
  { name: "Soil", color: "#c47f49", rgb: [196, 127, 73] },
  { name: "Bedrock", color: "#9ca5b4", rgb: [156, 165, 180] },
  { name: "Sand", color: "#e7c574", rgb: [231, 197, 116] },
  { name: "Big rock", color: "#82505e", rgb: [130, 80, 94] },
] as const;

export type Geometry = { x: number; y: number; width: number; height: number };
export function letterbox(width: number, height: number): Geometry {
  if (
    !Number.isFinite(width) ||
    !Number.isFinite(height) ||
    width <= 0 ||
    height <= 0
  )
    throw new Error("Image dimensions are invalid.");
  const scale = Math.min(SIZE / width, SIZE / height);
  const w = Math.max(1, Math.min(SIZE, Math.round(width * scale)));
  const h = Math.max(1, Math.min(SIZE, Math.round(height * scale)));
  return {
    x: Math.floor((SIZE - w) / 2),
    y: Math.floor((SIZE - h) / 2),
    width: w,
    height: h,
  };
}

export function imageKind(bytes: Uint8Array): "jpeg" | "png" | "webp" | null {
  if (
    bytes.length >= 3 &&
    bytes[0] === 0xff &&
    bytes[1] === 0xd8 &&
    bytes[2] === 0xff
  )
    return "jpeg";
  if (
    bytes.length >= 8 &&
    [137, 80, 78, 71, 13, 10, 26, 10].every((v, i) => bytes[i] === v)
  )
    return "png";
  if (
    bytes.length >= 12 &&
    String.fromCharCode(...bytes.slice(0, 4)) === "RIFF" &&
    String.fromCharCode(...bytes.slice(8, 12)) === "WEBP"
  )
    return "webp";
  return null;
}

export function classify(logits: Float32Array, region: Geometry) {
  if (logits.length !== SIZE * SIZE * 4 || !logits.every(Number.isFinite))
    throw new Error("The model returned an invalid result.");
  const pixels = SIZE * SIZE;
  const labels = new Uint8Array(pixels);
  const counts = [0, 0, 0, 0];
  for (let y = region.y; y < region.y + region.height; y++)
    for (let x = region.x; x < region.x + region.width; x++) {
      const p = y * SIZE + x;
      let best = 0;
      for (let c = 1; c < 4; c++)
        if (logits[c * pixels + p] > logits[best * pixels + p]) best = c;
      labels[p] = best;
      counts[best]++;
    }
  return { labels, counts, total: region.width * region.height };
}
