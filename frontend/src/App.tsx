import { useEffect, useRef, useState } from "react";
import type { ChangeEvent, DragEvent } from "react";
import {
  ArrowDown,
  ArrowRight,
  Check,
  ChevronRight,
  CircleHelp,
  Download,
  FlaskConical,
  ImagePlus,
  LoaderCircle,
  Upload,
  X,
} from "lucide-react";
import samples from "../public/samples/samples.json";
import expert from "../../results/expert-test-min3.json";
import { analyze, decodeFile } from "./lib/inference";
import type { Prediction, SourceImage } from "./lib/inference";
import { drawView } from "./lib/render";
import type { View } from "./lib/render";
import { CLASSES } from "./lib/terrain";

type Sample = (typeof samples)[number];

function CanvasView({
  source,
  prediction,
  view,
  enabled,
  opacity,
  label,
  className = "",
}: {
  source: SourceImage;
  prediction: Prediction | null;
  view: View;
  enabled: boolean[];
  opacity: number;
  label: string;
  className?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (ref.current)
      drawView(ref.current, source, prediction, view, enabled, opacity);
  }, [source, prediction, view, enabled, opacity]);
  return (
    <canvas ref={ref} className={className} role="img" aria-label={label} />
  );
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export default function App() {
  const [source, setSource] = useState<SourceImage | null>(null);
  const [sample, setSample] = useState<Sample | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [enabled, setEnabled] = useState([true, true, true, true]);
  const [opacity, setOpacity] = useState(0.65);
  const [compare, setCompare] = useState(50);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const request = useRef(0);

  async function chooseFile(file: File, chosen: Sample | null = null) {
    const id = ++request.current;
    setError("");
    setMessage("Opening image…");
    setBusy(true);
    try {
      const decoded = await decodeFile(file);
      if (id !== request.current) return;
      setSource({ ...decoded, sampleId: chosen?.id });
      setSample(chosen);
      setPrediction(null);
      setEnabled([true, true, true, true]);
      setMessage("");
    } catch (err) {
      if (id === request.current) {
        setError(
          err instanceof Error ? err.message : "Could not open this image.",
        );
        setMessage("");
      }
    } finally {
      if (id === request.current) setBusy(false);
    }
  }

  async function chooseSample(item: Sample) {
    setError("");
    setBusy(true);
    setMessage("Opening example…");
    try {
      const response = await fetch(
        `${import.meta.env.BASE_URL}${item.file.replace(/^\//, "")}`,
      );
      if (!response.ok)
        throw new Error("This example could not be loaded. Please try again.");
      const blob = await response.blob();
      await chooseFile(
        new File([blob], `${item.id}.jpg`, { type: blob.type || "image/jpeg" }),
        item,
      );
    } catch (err) {
      setBusy(false);
      setMessage("");
      setError(
        err instanceof Error ? err.message : "Could not load this example.",
      );
    }
  }

  async function run() {
    if (!source || busy) return;
    const id = ++request.current;
    setError("");
    setBusy(true);
    setMessage("Preparing…");
    try {
      const result = await analyze(source, (text) => {
        if (request.current === id) setMessage(text);
      });
      if (request.current === id) {
        setPrediction(result);
        setMessage("");
      }
    } catch (err) {
      if (request.current === id) {
        setError(
          err instanceof Error
            ? err.message
            : "Analysis failed. Please try again.",
        );
        setMessage("");
      }
    } finally {
      if (request.current === id) setBusy(false);
    }
  }

  function upload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) void chooseFile(file);
  }
  function drop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    if (!busy && event.dataTransfer.files[0])
      void chooseFile(event.dataTransfer.files[0]);
  }
  function download() {
    if (!source || !prediction) return;
    try {
      const canvas = document.createElement("canvas");
      drawView(canvas, source, prediction, "overlay", enabled, opacity);
      canvas.toBlob((blob) => {
        if (!blob) {
          setError("The image could not be downloaded.");
          return;
        }
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "marsterrain-overlay.png";
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }, "image/png");
    } catch {
      setError("The image could not be downloaded.");
    }
  }

  return (
    <>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="MarsTerrain home">
          <span className="brand-mark">
            <span />
          </span>
          <span>
            MARS<span className="brand-light">TERRAIN</span>
          </span>
        </a>
        <nav aria-label="Main navigation">
          <a className="nav-current" href="#lab">
            Analyze
          </a>
          <a href="#research">Research</a>
          <a
            className="nav-code"
            href="https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation"
            target="_blank"
            rel="noreferrer"
          >
            View source <ArrowRight size={14} />
          </a>
        </nav>
      </header>
      <main id="top">
        <section className="hero page-wrap">
          <div className="hero-copy">
            <h1>Mars Rover Terrain Segmentation</h1>
            <p>
              A computer vision model that identifies soil, bedrock, sand, and
              large rocks in Mars rover images.
            </p>
            <div className="hero-actions">
              <a className="primary-link" href="#lab">
                Try the model <ArrowDown size={16} />
              </a>
              <a className="text-link" href="#research">
                How it works <ArrowRight size={15} />
              </a>
            </div>
          </div>
          <div className="hero-art" aria-hidden="true">
            <div className="hero-image" />
            <div className="hero-art-overlay" />
          </div>
        </section>

        <section id="lab" className="lab page-wrap">
          <div className="section-heading">
            <div>
              <h2>Analyze an image</h2>
              <p>
                Choose an example or upload an image to see the predicted
                terrain.
              </p>
            </div>
            <div className="section-aside">
              Images are processed on your device.
            </div>
          </div>
          <div className="lab-shell">
            <div className="input-layout">
              <div className="sample-section">
                <div className="input-title">
                  <div>
                    <h3>Choose an example</h3>
                  </div>
                </div>
                <div className="sample-grid">
                  {samples.map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      className={`sample-tile ${sample?.id === item.id ? "selected" : ""}`}
                      aria-pressed={sample?.id === item.id}
                      disabled={busy}
                      onClick={() => void chooseSample(item)}
                    >
                      <img
                        src={`${import.meta.env.BASE_URL}${item.file.replace(/^\//, "")}`}
                        alt=""
                      />
                      <span className="tile-shade" />
                      {sample?.id === item.id && (
                        <span className="tile-check">
                          <Check size={14} />
                        </span>
                      )}
                      <span className="tile-title">
                        {item.title}
                        <ChevronRight size={15} />
                      </span>
                    </button>
                  ))}
                </div>
                <p className="source-note">
                  Curiosity Navcam images · NASA/JPL-Caltech / AI4Mars{" "}
                  <a href="#credits">View credits ↗</a>
                </p>
              </div>
              <div
                className={`upload-zone ${dragging ? "dragging" : ""}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  if (!busy) setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={drop}
              >
                <div className="upload-icon">
                  <Upload size={20} strokeWidth={1.5} />
                </div>
                <h3>Upload an image</h3>
                <p>Drop an image here, or choose a file from your device.</p>
                <input
                  ref={inputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={upload}
                  className="visually-hidden"
                  aria-label="Choose an image"
                  disabled={busy}
                />
                <button
                  type="button"
                  className="secondary-button"
                  disabled={busy}
                  onClick={() => inputRef.current?.click()}
                >
                  <ImagePlus size={15} /> Choose image
                </button>
                <small>
                  JPEG, PNG or WebP · up to 10 MB / 20 MP / 8,192 px per side
                </small>
              </div>
            </div>
            <div className="selection-bar">
              <div className="selection-info">
                <span
                  className={`selection-indicator ${source ? "ready" : ""}`}
                />{" "}
                <span>
                  {source ? (
                    <>
                      <strong>{sample?.title || source.name}</strong>
                      <small>
                        {source.width} × {source.height} px{" "}
                        {sample ? "· Curiosity Navcam" : "· Your image"}
                      </small>
                    </>
                  ) : (
                    <>
                      <strong>No image selected</strong>
                      <small>Choose an example or upload an image</small>
                    </>
                  )}
                </span>
              </div>
              <button
                type="button"
                className="analyze-button"
                disabled={!source || busy}
                onClick={() => void run()}
              >
                {busy ? (
                  <>
                    <LoaderCircle size={17} className="spin" /> Working…
                  </>
                ) : (
                  <>
                    Analyze terrain <ArrowRight size={17} />
                  </>
                )}
              </button>
            </div>
            {(message || error) && (
              <div
                className={`status-line ${error ? "error" : ""}`}
                role={error ? "alert" : "status"}
              >
                {error ? (
                  <>
                    <CircleHelp size={15} />
                    {error}
                    <button
                      type="button"
                      onClick={() => setError("")}
                      aria-label="Dismiss error"
                    >
                      <X size={15} />
                    </button>
                  </>
                ) : (
                  <>
                    <LoaderCircle size={15} className="spin" />
                    {message}
                  </>
                )}
              </div>
            )}
          </div>
          {source && (
            <div className="result-area">
              <div role="status" className="visually-hidden">
                {prediction ? "Analysis complete." : ""}
              </div>
              <div className="result-heading">
                <div>
                  <h3>{prediction ? "Results" : "Image ready"}</h3>
                  {prediction && (
                    <p>Predicted terrain classes for the selected image.</p>
                  )}
                </div>
                {prediction && (
                  <button
                    className="download-button"
                    onClick={download}
                    type="button"
                  >
                    <Download size={15} /> Download overlay
                  </button>
                )}
              </div>
              {!prediction && (
                <div className="preview-single">
                  <CanvasView
                    source={source}
                    prediction={null}
                    view="original"
                    enabled={enabled}
                    opacity={opacity}
                    label={`Selected image: ${source.name}`}
                  />
                  <div>
                    <span className="micro-label">SELECTED INPUT</span>
                    <strong>{sample?.title || source.name}</strong>
                    <p>
                      The first analysis downloads the model to your browser.
                    </p>
                  </div>
                </div>
              )}
              {prediction && (
                <>
                  <div className="image-triptych">
                    {(["original", "mask", "overlay"] as const).map((view) => (
                      <div className="image-card" key={view}>
                        <div className="image-card-title">
                          <span>
                            {view === "mask"
                              ? "PREDICTED MASK"
                              : view.toUpperCase()}
                          </span>
                        </div>
                        <div className="image-frame">
                          <CanvasView
                            source={source}
                            prediction={prediction}
                            view={view}
                            enabled={enabled}
                            opacity={opacity}
                            label={`${view} of ${source.name}`}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="result-controls">
                    <div className="legend-block">
                      <div className="control-heading">
                        <span>CLASS FILTER</span>
                      </div>
                      <div className="legend">
                        {CLASSES.map((item, i) => (
                          <button
                            type="button"
                            key={item.name}
                            aria-pressed={enabled[i]}
                            className={!enabled[i] ? "disabled-class" : ""}
                            onClick={() =>
                              setEnabled((old) =>
                                old.map((value, j) =>
                                  i === j ? !value : value,
                                ),
                              )
                            }
                          >
                            <span
                              className="legend-swatch"
                              style={{ background: item.color }}
                            />
                            {item.name}
                            <span className="legend-check">
                              {enabled[i] ? "●" : "○"}
                            </span>
                          </button>
                        ))}
                      </div>
                      <p>
                        The model assigns one of four classes to every pixel.
                      </p>
                    </div>
                    <div className="opacity-block">
                      <div className="control-heading">
                        <label htmlFor="opacity">OVERLAY OPACITY</label>
                        <span>{Math.round(opacity * 100)}%</span>
                      </div>
                      <input
                        id="opacity"
                        type="range"
                        min="0"
                        max="100"
                        value={Math.round(opacity * 100)}
                        onChange={(e) =>
                          setOpacity(Number(e.target.value) / 100)
                        }
                      />
                    </div>
                  </div>
                  <div className="analysis-grid">
                    <div className="comparison">
                      <div className="card-heading">
                        <div>
                          <h4>Compare original and overlay</h4>
                        </div>
                      </div>
                      <div className="compare-frame">
                        <CanvasView
                          source={source}
                          prediction={prediction}
                          view="original"
                          enabled={enabled}
                          opacity={opacity}
                          label="Original image in comparison"
                        />
                        <div
                          className="compare-overlay"
                          style={{ clipPath: `inset(0 ${100 - compare}% 0 0)` }}
                        >
                          <CanvasView
                            source={source}
                            prediction={prediction}
                            view="overlay"
                            enabled={enabled}
                            opacity={opacity}
                            label="Overlay in comparison"
                          />
                        </div>
                        <div
                          className="compare-handle"
                          style={{ left: `${compare}%` }}
                          aria-hidden="true"
                        >
                          <span>↔</span>
                        </div>
                        <input
                          id="compare"
                          className="compare-hit"
                          aria-label="Reveal overlay comparison"
                          type="range"
                          min="0"
                          max="100"
                          value={compare}
                          onChange={(e) => setCompare(Number(e.target.value))}
                        />
                      </div>
                    </div>
                    <div className="coverage">
                      <div className="card-heading">
                        <div>
                          <h4>Predicted coverage</h4>
                        </div>
                      </div>
                      <p>
                        Estimated share of image pixels, not model accuracy.
                      </p>
                      <div className="coverage-list">
                        {CLASSES.map((item, i) => (
                          <div className="coverage-row" key={item.name}>
                            <div>
                              <span>
                                <i style={{ background: item.color }} />
                                {item.name}
                              </span>
                              <strong>
                                {formatPercent(
                                  prediction.counts[i] / prediction.total,
                                )}
                              </strong>
                            </div>
                            <div className="bar-track">
                              <span
                                style={{
                                  width: `${(prediction.counts[i] / prediction.total) * 100}%`,
                                  background: item.color,
                                }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                      <div className="timing">
                        <span>MODEL INFERENCE</span>
                        <strong>{Math.round(prediction.elapsedMs)} ms</strong>
                      </div>
                    </div>
                  </div>
                  <div className="result-foot">
                    <span>
                      <Check size={14} /> Processed on this device
                    </span>
                    <span>
                      Model inference only · initial load{" "}
                      {Math.round(prediction.loadMs)} ms
                    </span>
                  </div>
                </>
              )}
              {sample && (
                <p className="selected-credit">
                  Image: {sample.sourceId} · {sample.credit}.{" "}
                  <a href="#credits">Source and attribution</a>
                </p>
              )}
            </div>
          )}
        </section>
        <section id="research" className="research">
          <div className="page-wrap">
            <div className="research-heading">
              <div>
                <h2>Model and results</h2>
              </div>
              <p>
                This is a research model, not a navigation system. It cannot
                determine whether terrain is safe to traverse.
              </p>
            </div>
            <div className="research-grid">
              <div className="research-story">
                <h3>How it works</h3>
                <p>
                  A compact model trained from scratch on AI4Mars Curiosity
                  images predicts a terrain class for each pixel. We checked the
                  data for overlap, selected the model on a validation split,
                  and evaluated it once against expert labels.
                </p>
                <a
                  href="https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation"
                  target="_blank"
                  rel="noreferrer"
                >
                  View code and methodology <ArrowRight size={15} />
                </a>
              </div>
              <div className="metric-card">
                <div className="metric-top">
                  <span>
                    <FlaskConical size={16} /> FROZEN EXPERT TEST
                  </span>
                  <span>AI4MARS / MIN3</span>
                </div>
                <div className="metric-main">
                  <strong>{formatPercent(expert.mean_iou)}</strong>
                  <span>
                    mean intersection
                    <br />
                    over union
                  </span>
                </div>
                <p>
                  Measured on {expert.evaluated_images} images with valid
                  labels, from {expert.total_images} expert images.{" "}
                  {expert.skipped.length} had no valid labeled pixels.
                </p>
                <div className="metric-classes">
                  {CLASSES.map((item, i) => (
                    <div key={item.name}>
                      <span>
                        <i style={{ background: item.color }} />
                        {item.name}
                      </span>
                      <strong>
                        {formatPercent(expert.per_class_iou[i] ?? 0)}
                      </strong>
                    </div>
                  ))}
                </div>
                <div className="metric-note">
                  Big rock has only {expert.support[3].toLocaleString()} labeled
                  pixels in this test; its estimate is especially uncertain.
                </div>
              </div>
            </div>
            <details className="research-detail">
              <summary>Evaluation details</summary>
              <div className="detail-grid">
                <div>
                  <h3>Test metrics</h3>
                  <p>
                    Pixel accuracy: {formatPercent(expert.pixel_accuracy)}. Mean
                    Dice: {formatPercent(expert.mean_dice)}.{" "}
                    {formatPercent(expert.ignored_fraction)} of pixels in
                    evaluated images were unlabeled and excluded.
                  </p>
                  <p>
                    The model may mislabel rover hardware, distant terrain,
                    unusual lighting, or images unlike its training data.
                  </p>
                  <span className="micro-label">CONFUSION MATRIX · PIXELS</span>
                  <p>
                    Rows are expert labels; columns are model predictions.
                    Ignored pixels are excluded.
                  </p>
                  <div className="table-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th scope="col">Reference</th>
                          {CLASSES.map((item) => (
                            <th scope="col" key={item.name}>
                              {item.name}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {expert.confusion_matrix.map((row, i) => (
                          <tr key={CLASSES[i].name}>
                            <th scope="row">{CLASSES[i].name}</th>
                            {row.map((value, j) => (
                              <td key={j}>{value.toLocaleString()}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
                <div>
                  <h3>Training and inference</h3>
                  <p>
                    The MobileNetV3-small model was trained in stages on up to
                    4,096 images and validated on 1,557 separate images. The
                    three training stages took 147, 322, and 467 seconds on MPS,
                    including validation and data loading.
                  </p>
                  <p>
                    The frozen ONNX model is 4.39 MB. Inputs are resized to 256
                    × 256; uploaded images are letterboxed to preserve their
                    shape. The model predicts one terrain class per pixel,
                    without calibrated confidence or a safety decision.
                  </p>
                  <img
                    className="curve-image"
                    src={`${import.meta.env.BASE_URL}research/validation-curves.png`}
                    alt="Validation mean IoU by epoch for the baseline, pretrained comparison, and scratch-trained experiments"
                  />
                  <small>
                    Validation-subset learning curves. Continuation runs start
                    from earlier scratch checkpoints.
                  </small>
                </div>
              </div>
            </details>
          </div>
        </section>
      </main>
      <footer id="credits">
        <div className="page-wrap footer-main">
          <div>
            <a className="brand" href="#top">
              <span className="brand-mark">
                <span />
              </span>
              <span>
                MARS<span className="brand-light">TERRAIN</span>
              </span>
            </a>
            <p>
              An independent research demonstration.
              <br />
              Not affiliated with or endorsed by NASA.
            </p>
          </div>
          <div>
            <span className="micro-label">DATA & RESEARCH</span>
            <a
              href="https://openaccess.thecvf.com/content/CVPR2021W/AI4Space/html/Swan_AI4MARS_A_Dataset_for_Terrain-Aware_Autonomous_Driving_on_Mars_CVPRW_2021_paper.html"
              target="_blank"
              rel="noreferrer"
            >
              Swan et al. · AI4Mars ↗
            </a>
            <a
              href="https://zenodo.org/records/15995036"
              target="_blank"
              rel="noreferrer"
            >
              NASA / JPL-Caltech data ↗
            </a>
            <a
              href="https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation"
              target="_blank"
              rel="noreferrer"
            >
              Code & model documentation ↗
            </a>
          </div>
        </div>
        <div className="page-wrap footer-bottom">
          <span>MARS TERRAIN SEGMENTATION / OPEN RESEARCH</span>
          <span>MODEL, DATA AND CODE</span>
        </div>
      </footer>
    </>
  );
}
