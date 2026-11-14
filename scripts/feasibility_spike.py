"""Exercise a real training record and ONNX CPU; output is not a trained model."""
import gzip
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'data' / 'processed' / 'spike'


def main():
    torch.manual_seed(42)
    manifest = ROOT / 'data' / 'manifests' / 'image_manifest.jsonl.gz'
    with gzip.open(manifest, 'rt') as stream:
        eligible = {row['row'] for line in stream if (row := json.loads(line))['shard'] == 'train-00000-of-00013.parquet' and row['assigned_split'] == 'train'}
    shard = pq.ParquetFile(RAW / 'train-00000-of-00013.parquet')
    sample = None
    offset = 0
    for batch in shard.iter_batches(batch_size=8):
        for row in batch.to_pylist():
            index = offset
            offset += 1
            if index not in eligible or not row['has_labels'] or not row['has_masks']:
                continue
            label = np.array(Image.open(io.BytesIO(row['label_mask']['bytes'])))
            rover = np.array(Image.open(io.BytesIO(row['rover_mask']['bytes'])))
            distance = np.array(Image.open(io.BytesIO(row['range_mask']['bytes'])))
            if label.ndim != 2 or not set(np.unique(label)).issubset({0, 1, 2, 3, 255}) or label.shape != rover.shape or label.shape != distance.shape:
                raise ValueError('Unexpected mask format')
            label[(rover == 1) | (distance == 1)] = 255
            if np.any(label != 255):
                sample = row, label
                break
        if sample is not None:
            break
    if sample is None:
        raise ValueError('No eligible sample')
    row, label = sample
    image = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)
    mask = Image.fromarray(label).resize((256, 256), Image.Resampling.NEAREST)
    x = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().unsqueeze(0) / 255
    y = torch.from_numpy(np.asarray(mask).copy().astype(np.int64)).unsqueeze(0)
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 4, 1))
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = F.cross_entropy(model(x), y, ignore_index=255)
    optim.zero_grad(); loss.backward(); optim.step()
    model.eval()
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'untrained_spike.onnx'
    torch.onnx.export(model, x, str(target), input_names=['image'], output_names=['logits'], opset_version=17, dynamo=False)
    with torch.no_grad():
        expected = model(x).numpy()
    actual = ort.InferenceSession(str(target), providers=['CPUExecutionProvider']).run(None, {'image': x.numpy()})[0]
    error = float(np.max(np.abs(expected - actual)))
    if error > 1e-4:
        raise AssertionError(f'ONNX mismatch: {error}')
    palette = np.array([[196, 127, 73], [156, 165, 180], [231, 197, 116], [130, 80, 94]], dtype=np.uint8)
    Image.fromarray(palette[expected.argmax(axis=1)[0]]).save(OUT / 'untrained_spike_mask.png')
    print(json.dumps({'sample_id': row['image']['path'], 'loss': float(loss.detach()), 'valid_pixels': int((y != 255).sum()), 'max_abs_error': error, 'onnx_bytes': target.stat().st_size, 'warning': 'single update, not a trained model'}, indent=2))


if __name__ == '__main__':
    main()
