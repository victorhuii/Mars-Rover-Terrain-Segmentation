function resizeWeights(input: number, output: number) {
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
  outputWidth: number,
  outputHeight: number,
) {
  if (
    rgba.length !== width * height * 4 ||
    Math.min(width, height, outputWidth, outputHeight) <= 0
  )
    throw new Error("Invalid image dimensions.");
  const horizontal = resizeWeights(width, outputWidth);
  const vertical = resizeWeights(height, outputHeight);
  const intermediate = new Uint8Array(outputWidth * height * 3);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < outputWidth; x++) {
      const base = (y * outputWidth + x) * 3;
      for (let channel = 0; channel < 3; channel++) {
        let value = 0;
        for (const contribution of horizontal[x])
          value +=
            rgba[(y * width + contribution.index) * 4 + channel] *
            contribution.weight;
        intermediate[base + channel] = Math.min(255, Math.floor(value + 0.5));
      }
    }
  const output = new Uint8Array(outputWidth * outputHeight * 3);
  for (let y = 0; y < outputHeight; y++)
    for (let x = 0; x < outputWidth; x++) {
      const base = (y * outputWidth + x) * 3;
      for (let channel = 0; channel < 3; channel++) {
        let value = 0;
        for (const contribution of vertical[y])
          value +=
            intermediate[(contribution.index * outputWidth + x) * 3 + channel] *
            contribution.weight;
        output[base + channel] = Math.min(255, Math.floor(value + 0.5));
      }
    }
  return output;
}
