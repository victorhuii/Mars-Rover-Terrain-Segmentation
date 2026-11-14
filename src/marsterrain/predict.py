"""Local ONNX inference using the same letterboxing and palette as the web app."""
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from .dataset import CLASSES
from .evaluate import sha256

MODEL = Path(__file__).resolve().parents[2] / 'frontend/public/models/marsterrain-v1-be2513a68f69.onnx'


@dataclass(frozen=True)
class Prediction:
    mask: Image.Image
    coverage: dict[str, float]
    inference_ms: float


@lru_cache(maxsize=2)
def load_session(model: str):
    path = Path(model); metadata = json.loads(path.with_suffix('.json').read_text())
    if sha256(path) != metadata['onnx_sha256']: raise ValueError('Model hash does not match metadata')
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def segment_image(image: Image.Image, model: str | Path = MODEL) -> Prediction:
    image = ImageOps.exif_transpose(image)
    if image.width < 1 or image.height < 1 or image.width * image.height > 20_000_000 or max(image.size) > 8192:
        raise ValueError('Image exceeds the 20-megapixel or 8,192-pixel side limit')
    if 'A' in image.getbands():
        rgba = image.convert('RGBA'); base = Image.new('RGB', rgba.size, 'black'); base.paste(rgba, mask=rgba.getchannel('A')); image = base
    else: image = image.convert('RGB')
    scale = min(256 / image.width, 256 / image.height)
    width = max(1, min(256, math.floor(image.width * scale + 0.5)))
    height = max(1, min(256, math.floor(image.height * scale + 0.5)))
    left, top = (256 - width) // 2, (256 - height) // 2
    canvas = Image.new('RGB', (256, 256), 'black')
    canvas.paste(image.resize((width, height), Image.Resampling.BILINEAR), (left, top))
    tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255
    session = load_session(str(Path(model).resolve()))
    start = time.perf_counter(); logits = session.run(['logits'], {'image': tensor})[0]; elapsed = (time.perf_counter() - start) * 1000
    if logits.shape != (1, 4, 256, 256) or not np.isfinite(logits).all(): raise ValueError('Invalid model output')
    labels = logits.argmax(axis=1)[0, top:top + height, left:left + width].astype(np.uint8)
    counts = np.bincount(labels.ravel(), minlength=4)
    coverage = {name: float(counts[i] / labels.size) for i, name in enumerate(CLASSES)}
    mask = Image.fromarray(labels).resize(image.size, Image.Resampling.NEAREST)
    return Prediction(mask=mask, coverage=coverage, inference_ms=elapsed)
