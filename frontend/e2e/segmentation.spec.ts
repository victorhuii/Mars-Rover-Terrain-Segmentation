import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

test("sample runs the trained model, controls change the mask and overlay downloads", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const requests: string[] = [];
  const failedResponses: string[] = [];
  page.on("response", (response) => {
    if (response.status() >= 400)
      failedResponses.push(`${response.status()} ${response.url()}`);
  });
  page.on("request", (request) => requests.push(request.url()));
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: /Mars Rover Terrain Segmentation/i }),
  ).toBeVisible();
  await page.getByRole("button", { name: /Open plain/i }).click();
  await expect(page.getByText("Image ready")).toBeVisible();
  await page.getByRole("button", { name: /Analyze terrain/i }).click();
  await expect(
    page.getByRole("heading", { name: "Results", exact: true }),
  ).toBeVisible();
  expect(
    requests.some((url) => url.includes("marsterrain-v1-be2513a68f69.onnx")),
  ).toBeTruthy();
  await expect(
    page.getByRole("heading", { name: "Predicted coverage" }),
  ).toBeVisible();
  const mask = page.getByRole("img", { name: /mask of/i });
  const before = await mask.evaluate((canvas: HTMLCanvasElement) =>
    canvas.toDataURL(),
  );
  const reference = await readFile(resolve("e2e/fixtures/plain-reference.bin"));
  const browserLabels = await mask.evaluate((canvas: HTMLCanvasElement) => {
    const context = canvas.getContext("2d")!;
    const data = context.getImageData(0, 0, canvas.width, canvas.height).data;
    const colors = [
      [196, 127, 73],
      [156, 165, 180],
      [231, 197, 116],
      [130, 80, 94],
    ];
    const labels: number[] = [];
    for (let y = 0; y < 256; y++)
      for (let x = 0; x < 256; x++) {
        const offset =
          (Math.floor((y * canvas.height) / 256) * canvas.width +
            Math.floor((x * canvas.width) / 256)) *
          4;
        labels.push(
          colors.findIndex((color) =>
            color.every((value, c) => value === data[offset + c]),
          ),
        );
      }
    return labels;
  });
  expect(browserLabels.length).toBe(reference.length);
  expect(
    browserLabels.filter((value, index) => value === reference[index]).length /
      reference.length,
  ).toBeGreaterThanOrEqual(0.995);
  for (const name of ["Soil", "Bedrock", "Sand", "Big rock"])
    await page.getByRole("button", { name: new RegExp(`^${name}`) }).click();
  const after = await mask.evaluate((canvas: HTMLCanvasElement) =>
    canvas.toDataURL(),
  );
  expect(after).not.toBe(before);
  await page.getByRole("button", { name: /^Soil/ }).click();
  await page.getByLabel("OVERLAY OPACITY").fill("25");
  await expect(page.getByText("25%", { exact: true })).toBeVisible();
  await page.getByLabel("Reveal overlay comparison").fill("70");
  await expect(page.getByLabel("Reveal overlay comparison")).toHaveValue("70");
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: /Download overlay/i }).click();
  const download = await downloadPromise;
  const file = await download.path();
  expect(file).not.toBeNull();
  const bytes = await readFile(file!);
  expect([...bytes.subarray(0, 8)]).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
  await page.getByText("Evaluation details").click();
  await expect(page.getByRole("table")).toBeVisible();
  const accessibility = await new AxeBuilder({ page }).analyze();
  expect(accessibility.violations).toEqual([]);
  expect(errors).toEqual([]);
  expect(failedResponses).toEqual([]);
});

test("mobile upload honors EXIF orientation, validates files, and performs inference", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  const input = page.getByLabel("Choose an image");
  await input.setInputFiles({
    name: "broken.png",
    mimeType: "image/png",
    buffer: Buffer.from("not an image"),
  });
  await expect(page.getByRole("alert")).toContainText("not a supported");
  await input.setInputFiles({
    name: "large.jpg",
    mimeType: "image/jpeg",
    buffer: Buffer.alloc(10 * 1024 * 1024 + 1),
  });
  await expect(page.getByRole("alert")).toContainText("larger than 10 MB");
  const oversizedDimension = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 8193;
    canvas.height = 1;
    return canvas.toDataURL("image/png").split(",")[1];
  });
  await input.setInputFiles({
    name: "wide.png",
    mimeType: "image/png",
    buffer: Buffer.from(oversizedDimension, "base64"),
  });
  await expect(page.getByRole("alert")).toContainText("8,192-pixel side limit");
  await input.setInputFiles(resolve("e2e/fixtures/oriented.jpg"));
  await expect(page.getByText("90 × 160 px")).toBeVisible();
  await page.getByRole("button", { name: /Analyze terrain/i }).focus();
  await page.keyboard.press("Enter");
  await expect(
    page.getByRole("heading", { name: "Results", exact: true }),
  ).toBeVisible();
  const canvas = page.getByRole("img", { name: /original of/i }).first();
  await expect(canvas).toBeVisible();
  const dimensions = await canvas.evaluate((node: HTMLCanvasElement) => [
    node.width,
    node.height,
  ]);
  expect(dimensions).toEqual([90, 160]);
  await expect(
    page.getByRole("button", { name: /Download overlay/i }),
  ).toBeVisible();
});
