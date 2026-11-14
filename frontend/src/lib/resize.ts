/** Antialiased separable bilinear resize, matching Pillow's pixel-center geometry. */
function weights(input: number, output: number) {
  const scale = input / output;
  const support = Math.max(1, scale);
  return Array.from({ length: output }, (_, index) => {
    const center = (index + 0.5) * scale;
    const start = Math.max(0, Math.floor(center - support));
    const end = Math.min(input, Math.ceil(center + support));
    const values: { index: number; weight: number }[] = [];
    let total = 0;
    for (let i = start; i < end; i++) {
      const weight = Math.max(0, 1 - Math.abs((i + 0.5 - center) / support));
      if (weight) {
        values.push({ index: i, weight });
        total += weight;
      }
    }
    return values.map((value) => ({
      index: value.index,
      weight: value.weight / total,
    }));
  });
}

export function resizeRgb(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  outWidth: number,
  outHeight: number,
) {
  if (
    rgba.length !== width * height * 4 ||
    Math.min(width, height, outWidth, outHeight) <= 0
  )
    throw new Error("Invalid image dimensions.");
  const horizontal = weights(width, outWidth);
  const vertical = weights(height, outHeight);
  const intermediate = new Uint8Array(outWidth * height * 3);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < outWidth; x++) {
      const base = (y * outWidth + x) * 3;
      for (let c = 0; c < 3; c++) {
        let value = 0;
        for (const part of horizontal[x])
          value += rgba[(y * width + part.index) * 4 + c] * part.weight;
        intermediate[base + c] = Math.min(255, Math.floor(value + 0.5));
      }
    }
  const output = new Uint8Array(outWidth * outHeight * 3);
  for (let y = 0; y < outHeight; y++)
    for (let x = 0; x < outWidth; x++) {
      const base = (y * outWidth + x) * 3;
      for (let c = 0; c < 3; c++) {
        let value = 0;
        for (const part of vertical[y])
          value +=
            intermediate[(part.index * outWidth + x) * 3 + c] * part.weight;
        output[base + c] = Math.min(255, Math.floor(value + 0.5));
      }
    }
  return output;
}
