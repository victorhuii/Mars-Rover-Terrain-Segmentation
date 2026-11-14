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
  ScanLine,
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
            Terrain lab
          </a>
          <a href="#research">The research</a>
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
            <div className="eyebrow">
              <span className="pulse-dot" /> A COMPUTER VISION RESEARCH DEMO{" "}
              <span className="eyebrow-line" /> 01 / TERRAIN LAB
            </div>
            <h1>
              See Mars through
              <br />
              <em>a rover’s eyes.</em>
            </h1>
            <p>
              Explore how a trained model identifies terrain in historical rover
              imagery, one pixel at a time.
            </p>
            <div className="hero-actions">
              <a className="primary-link" href="#lab">
                Explore the terrain <ArrowDown size={16} />
              </a>
              <a className="text-link" href="#research">
                How it works <ArrowRight size={15} />
              </a>
            </div>
          </div>
          <div className="hero-art" aria-hidden="true">
            <div className="hero-image" />
            <div className="hero-art-overlay" />
            <div className="hero-art-top">
              <span>MSL / NAVCAM</span>
              <span>IMAGE STUDY 001</span>
            </div>
            <div className="hero-art-bottom">
              <span>
                GALE CRATER
                <br />
                <strong>MARS</strong>
              </span>
              <span className="art-cross">✳</span>
            </div>
          </div>
          <div className="hero-foot">
            <span>
              <span className="foot-label">MISSION</span> CURIOSITY / MSL
            </span>
            <span>
              <span className="foot-label">DATA</span> AI4MARS
            </span>
            <span>
              <span className="foot-label">METHOD</span> SEMANTIC SEGMENTATION
            </span>
            <span className="foot-index">SCROLL TO EXPLORE ↓</span>
          </div>
        </section>

        <section id="lab" className="lab page-wrap">
          <div className="section-heading">
            <div>
              <div className="eyebrow">
                <span className="index">01</span> THE INTERACTIVE LAB
              </div>
              <h2>
                Read the landscape<span className="period">.</span>
              </h2>
              <p>
                Choose a rover image or bring your own. The model runs right
                here in your browser.
              </p>
            </div>
            <div className="section-aside">
              <span className="aside-dot" /> PRIVATE BY DESIGN{" "}
              <small>Your uploads stay on your device.</small>
            </div>
          </div>
          <div className="lab-shell">
            <div className="lab-topline">
              <span>
                <span className="tiny-square" /> INPUT / IMAGE SELECTION
              </span>
              <span>
                STEP 01 <span className="muted">—</span> 02
              </span>
            </div>
            <div className="input-layout">
              <div className="sample-section">
                <div className="input-title">
                  <div>
                    <span className="micro-label">THE ARCHIVE</span>
                    <h3>Start with a scene</h3>
                  </div>
                  <span className="count-badge">04 EXAMPLES</span>
                </div>
                <div className="sample-grid">
                  {samples.map((item, index) => (
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
                      <span className="tile-number">0{index + 1}</span>
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
                <span className="micro-label">YOUR OWN IMAGE</span>
                <h3>Bring a landscape</h3>
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
                      <small>
                        Choose an example or upload an image to begin
                      </small>
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
          {!source && (
            <div className="empty-result">
              <div className="empty-icon">
                <ScanLine size={23} strokeWidth={1.2} />
              </div>
              <div>
                <strong>Your analysis will appear here</strong>
                <p>
                  Select an image above to reveal a pixel-by-pixel view of the
                  terrain.
                </p>
              </div>
              <span className="empty-index">AWAITING INPUT / 02</span>
            </div>
          )}
          {source && (
            <div className="result-area">
              <div className="result-heading">
                <div>
                  <div className="eyebrow">
                    <span className="index">02</span>{" "}
                    {prediction ? "ANALYSIS COMPLETE" : "READY TO ANALYZE"}
                  </div>
                  <h3>
                    {prediction ? "A closer look" : "Your image is ready"}
                  </h3>
                  <p>
                    {prediction
                      ? "Explore the model’s predicted labels and compare them with the original."
                      : "Run the model to see the predicted terrain mask and overlay."}
                  </p>
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
                      Analysis runs locally. The first run downloads the model
                      to your browser.
                    </p>
                  </div>
                </div>
              )}
              {prediction && (
                <>
                  <div className="image-triptych">
                    {(["original", "mask", "overlay"] as const).map(
                      (view, index) => (
                        <div className="image-card" key={view}>
                          <div className="image-card-title">
                            <span>
                              0{index + 1} /{" "}
                              {view === "mask"
                                ? "PREDICTED MASK"
                                : view.toUpperCase()}
                            </span>
                            <span>
                              {view === "original" ? "INPUT" : "MODEL OUTPUT"}
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
                      ),
                    )}
                  </div>
                  <div className="result-controls">
                    <div className="legend-block">
                      <div className="control-heading">
                        <span>CLASS FILTER</span>
                        <span>CLICK TO TOGGLE</span>
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
                        Unknown / unlabeled pixels exist in reference masks;
                        this model always predicts one of the four classes.
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
                          <span className="micro-label">BEFORE / AFTER</span>
                          <h4>Compare the view</h4>
                        </div>
                        <span>DRAG TO REVEAL</span>
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
                          <span className="micro-label">MODEL ESTIMATE</span>
                          <h4>Predicted coverage</h4>
                        </div>
                        <span>IMAGE AREA</span>
                      </div>
                      <p>
                        Share of the image assigned to each class. These are
                        pixel estimates, not accuracy or calibrated confidence.
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
                <div className="eyebrow">
                  <span className="index">03</span> UNDER THE HOOD
                </div>
                <h2>
                  Built for understanding,
                  <br />
                  <em>not navigation.</em>
                </h2>
              </div>
              <p>
                Terrain labels can provide context for future rover autonomy.
                This demonstration does not determine whether a route is safe or
                traversable.
              </p>
            </div>
            <div className="research-grid">
              <div className="research-story">
                <span className="micro-label">THE QUESTION</span>
                <h3>What can a rover image tell us?</h3>
                <p>
                  Loose soil, sand, exposed bedrock and large rocks interact
                  differently with rover wheels. AI4Mars pairs historical images
                  with terrain labels so researchers can study pixel-level
                  classification.
                </p>
                <p>
                  We trained a compact segmentation network from scratch,
                  selected it using a separate validation split, and evaluated
                  it once on expert labels. The interactive lab runs that
                  trained model in your browser.
                </p>
                <a
                  href="https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation"
                  target="_blank"
                  rel="noreferrer"
                >
                  Explore the methodology <ArrowRight size={15} />
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
              <summary>
                Explore the evaluation{" "}
                <span>METRICS, METHOD & LIMITATIONS</span>
              </summary>
              <div className="detail-grid">
                <div>
                  <span className="micro-label">THE HELD-OUT RESULTS</span>
                  <h3>What the numbers describe</h3>
                  <p>
                    The model was selected on a separate validation split before
                    one expert test. On {expert.evaluated_images} test images
                    containing valid labels, it achieved{" "}
                    {formatPercent(expert.pixel_accuracy)} pixel accuracy and{" "}
                    {formatPercent(expert.mean_dice)} mean Dice.{" "}
                    {formatPercent(expert.ignored_fraction)} of pixels in
                    evaluated images were ignored; {expert.skipped.length} other
                    images contained no valid labeled pixels.
                  </p>
                  <p>
                    Big-rock IoU is{" "}
                    {formatPercent(expert.per_class_iou[3] ?? 0)} and its
                    support is only {expert.support[3].toLocaleString()} pixels.
                    A prediction on an unlabeled pixel is not measured by these
                    metrics. The model may mislabel rover hardware, distant
                    terrain, unusual lighting or images outside the training
                    distribution.
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
                  <span className="micro-label">REPRODUCIBLE BY DESIGN</span>
                  <h3>From images to inference</h3>
                  <p>
                    The selected scratch-trained MobileNetV3-small encoder and
                    compact decoder were trained in stages on 512, 2,048 and
                    4,096 audited training images. The complete clean split
                    contains 9,154 training and 1,557 validation images. The
                    three training stages took about 147, 322 and 467 seconds
                    respectively on MPS, including validation and data loading.
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
            <div className="process-strip">
              <div>
                <span>01 / DATA</span>
                <strong>Rover imagery</strong>
                <small>AI4Mars Curiosity subset</small>
              </div>
              <ArrowRight size={16} />
              <div>
                <span>02 / LEARN</span>
                <strong>Pixel-level model</strong>
                <small>Scratch-trained compact network</small>
              </div>
              <ArrowRight size={16} />
              <div>
                <span>03 / EXPLORE</span>
                <strong>Browser inference</strong>
                <small>ONNX Runtime Web / WASM</small>
              </div>
            </div>
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
          <div className="footer-note">
            <span>RESEARCH PROTOTYPE</span>
            <p>
              Semantic classification only. Not a rover navigation or safety
              system.
            </p>
          </div>
        </div>
        <div className="page-wrap footer-bottom">
          <span>MARS TERRAIN SEGMENTATION / OPEN RESEARCH</span>
          <span>MADE FOR CURIOSITY, NOT MISSION CONTROL</span>
        </div>
      </footer>
    </>
  );
}
