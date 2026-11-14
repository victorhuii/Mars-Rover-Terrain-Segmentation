import { describe, expect, it } from "vitest";
import { resizeRgb } from "./resize";

describe("bilinear image resize", () => {
  it("preserves constant RGB values for up- and down-sampling", () => {
    const pixels = new Uint8ClampedArray(4 * 4 * 4);
    for (let i = 0; i < 16; i++) pixels.set([10, 90, 200, 255], i * 4);
    expect([...resizeRgb(pixels, 4, 4, 2, 2)]).toEqual(
      Array.from({ length: 4 }, () => [10, 90, 200]).flat(),
    );
    expect([...resizeRgb(pixels, 4, 4, 8, 8)].slice(0, 3)).toEqual([
      10, 90, 200,
    ]);
  });
  it("matches a Pillow bilinear reference for a non-integer downscale", () => {
    const pixels = new Uint8ClampedArray(5 * 4 * 4);
    for (let i = 0; i < 20; i++)
      pixels.set([i * 12, i * 12 + 4, i * 12 + 8, 255], i * 4);
    expect([...resizeRgb(pixels, 5, 4, 3, 2)]).toEqual([
      48, 52, 56, 67, 71, 75, 86, 90, 94, 142, 146, 150, 161, 165, 169, 180,
      184, 188,
    ]);
  });
  it("rejects inconsistent source dimensions", () => {
    expect(() => resizeRgb(new Uint8ClampedArray(4), 2, 2, 1, 1)).toThrow();
  });
});
