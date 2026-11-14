import { CLASSES, SIZE } from "./terrain";
import type { Prediction, SourceImage } from "./inference";

export type View = "original" | "mask" | "overlay";
export function drawView(
  canvas: HTMLCanvasElement,
  source: SourceImage,
  prediction: Prediction | null,
  view: View,
  enabled: boolean[],
  opacity: number,
) {
  const scale = Math.min(1, 2048 / Math.max(source.width, source.height));
  canvas.width = Math.max(1, Math.round(source.width * scale));
  canvas.height = Math.max(1, Math.round(source.height * scale));
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Your browser could not render the result.");
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (view !== "mask")
    context.drawImage(source.image, 0, 0, canvas.width, canvas.height);
  if (!prediction || view === "original") return;
  const mask = document.createElement("canvas");
  mask.width = SIZE;
  mask.height = SIZE;
  const maskContext = mask.getContext("2d");
  if (!maskContext) throw new Error("Your browser could not render the mask.");
  const image = maskContext.createImageData(SIZE, SIZE);
  const { region, labels } = prediction;
  for (let y = region.y; y < region.y + region.height; y++)
    for (let x = region.x; x < region.x + region.width; x++) {
      const p = y * SIZE + x;
      const label = labels[p];
      const color = CLASSES[label].rgb;
      image.data[p * 4] = color[0];
      image.data[p * 4 + 1] = color[1];
      image.data[p * 4 + 2] = color[2];
      image.data[p * 4 + 3] = enabled[label] ? 255 : 0;
    }
  maskContext.putImageData(image, 0, 0);
  context.imageSmoothingEnabled = false;
  context.globalAlpha = view === "overlay" ? opacity : 1;
  context.drawImage(
    mask,
    region.x,
    region.y,
    region.width,
    region.height,
    0,
    0,
    canvas.width,
    canvas.height,
  );
  context.globalAlpha = 1;
}
