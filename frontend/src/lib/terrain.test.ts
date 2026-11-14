import { describe, expect, it } from "vitest";
import { classify, imageKind, letterbox, SIZE } from "./terrain";

describe("image validation and model geometry", () => {
  it("recognizes actual JPEG, PNG, and WebP signatures", () => {
    expect(imageKind(new Uint8Array([255, 216, 255, 0]))).toBe("jpeg");
    expect(imageKind(new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]))).toBe(
      "png",
    );
    expect(
      imageKind(new Uint8Array([82, 73, 70, 70, 0, 0, 0, 0, 87, 69, 66, 80])),
    ).toBe("webp");
    expect(imageKind(new Uint8Array([1, 2, 3]))).toBeNull();
  });
  it("preserves aspect ratio and centers padding", () => {
    expect(letterbox(1200, 600)).toEqual({
      x: 0,
      y: 64,
      width: 256,
      height: 128,
    });
    expect(letterbox(100, 200)).toEqual({
      x: 64,
      y: 0,
      width: 128,
      height: 256,
    });
    expect(() => letterbox(0, 10)).toThrow();
  });
  it("counts only content pixels and assigns the maximum-logit class", () => {
    const logits = new Float32Array(4 * SIZE * SIZE);
    logits.fill(1, SIZE * SIZE, 2 * SIZE * SIZE);
    const result = classify(logits, { x: 2, y: 3, width: 4, height: 5 });
    expect(result.counts).toEqual([0, 20, 0, 0]);
    expect(result.total).toBe(20);
    expect(result.labels[3 * SIZE + 2]).toBe(1);
    logits[0] = Number.NaN;
    expect(() =>
      classify(logits, { x: 0, y: 0, width: 1, height: 1 }),
    ).toThrow();
  });
});
