"""One-shot expert evaluation after a checkpoint and protocol are frozen.

This command refuses to read expert annotation columns until all frozen hashes
and source image identities pass. Its exclusive result file prevents accidental
re-evaluation with another model. Only use trusted local checkpoints.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw

from .dataset import decode_image, prepare_pair
from .evaluate import sha256
from .metrics import confusion_matrix, summarize
from .models import make_model

EXPECTED_EXPERT_COUNT = 322


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=Path('data/raw'))
    parser.add_argument('--manifests', type=Path, default=Path('data/manifests'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--qualitative', type=Path, help='Save local best and worst expert examples after evaluation')
    args = parser.parse_args()
    frozen = json.loads(args.freeze.read_text())
    if frozen.get('split') != 'test_min3' or frozen.get('protocol') != 'ai4mars-expert-min3-v1':
        raise ValueError('Invalid frozen test protocol')
    audit = args.manifests / 'image_manifest.jsonl.gz'
    prepared = args.manifests / 'prepared_manifest.json.gz'
    shard = args.raw / 'test_min3-00000-of-00001.parquet'
    for path, field in ((args.checkpoint, 'checkpoint_sha256'), (audit, 'audit_manifest_sha256'),
                        (prepared, 'prepared_manifest_gzip_sha256'), (shard, 'expert_shard_sha256')):
        if sha256(path) != frozen[field]: raise ValueError(f'Frozen hash mismatch: {field}')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['metadata']['overfit_validation_is_training']:
        raise ValueError('Cannot evaluate an overfit diagnostic')
    if checkpoint['metadata']['prepared_manifest_sha256'] != frozen['prepared_manifest_sha256']:
        raise ValueError('Checkpoint dataset mismatch')
    with gzip.open(audit, 'rt') as f:
        expected = {r['row']: r for line in f if (r := json.loads(line))['source_split'] == 'test_min3'}
    parquet = pq.ParquetFile(shard)
    if len(expected) != EXPECTED_EXPERT_COUNT or parquet.metadata.num_rows != len(expected):
        raise ValueError('Unexpected expert image count')
    # Verify every image identity without opening the expert annotation columns.
    index = 0
    for batch in parquet.iter_batches(batch_size=8, columns=['image']):
        for item in batch.to_pylist():
            image = item['image']; row = expected[index]
            if image['path'] != row['id'] or hashlib.sha256(image['bytes']).hexdigest() != row['byte_sha256']:
                raise ValueError(f'Expert image mismatch at row {index}')
            index += 1
    if args.device == 'mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable')
    config = checkpoint['metadata']['config']; device = torch.device(args.device)
    model = make_model(config.get('architecture', 'unet'), channels=config['channels'], pretrained=False).to(device)
    model.load_state_dict(checkpoint['model']); model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation remains if evaluation fails: investigation is required
    # before any rerun so that incomplete test access is not silently forgotten.
    descriptor = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, 'w') as output:
        output.write(json.dumps({'status': 'started', 'freeze_sha256': sha256(args.freeze)}) + '\n')
    matrix = np.zeros((4, 4), dtype=np.int64)
    pixels = 0; processed = 0; skipped = []; per_image = []; extremes = []; index = 0; start = time.monotonic()
    with torch.no_grad():
        for batch in parquet.iter_batches(batch_size=4, columns=['image', 'label_mask', 'rover_mask', 'range_mask']):
            for row in batch.to_pylist():
                try:
                    if any(not row[k] or not row[k].get('bytes') for k in ('image', 'label_mask', 'rover_mask', 'range_mask')):
                        raise ValueError('missing image or required mask')
                    image, mask = prepare_pair(*(decode_image(row[k]['bytes']) for k in ('image', 'label_mask', 'rover_mask', 'range_mask')), size=config['size'])
                except (OSError, ValueError) as error:
                    skipped.append({'row': index, 'id': expected[index]['id'], 'reason': str(error)})
                    index += 1
                    continue
                x = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().unsqueeze(0).div_(255).to(device)
                target = np.asarray(mask); prediction = model(x).argmax(1)[0].cpu().numpy()
                sample_matrix = confusion_matrix(target, prediction)
                matrix += sample_matrix; pixels += target.size; processed += 1
                sample_result = summarize(sample_matrix)
                per_image.append({'id': expected[index]['id'], 'valid_pixels': sample_result['valid_pixels'], 'mean_iou': sample_result['mean_iou']})
                if args.qualitative and sample_result['mean_iou'] is not None:
                    extremes.append((sample_result['mean_iou'], expected[index]['id'], np.asarray(image).copy(), target.copy(), prediction.copy()))
                    extremes.sort(key=lambda entry: entry[0])
                    if len(extremes) > 4: extremes = extremes[:2] + extremes[-2:]
                index += 1
    if processed == 0 or matrix.sum() == 0:
        raise ValueError('No valid expert pixels; the result reservation remains for investigation')
    result = summarize(matrix)
    if args.qualitative and extremes:
        size = config['size']; palette = np.array([[196,127,73], [156,165,180], [231,197,116], [130,80,94]], dtype=np.uint8)
        canvas = Image.new('RGB', (size * 4, (size + 24) * len(extremes))); draw = ImageDraw.Draw(canvas)
        for i, (score, ident, image, label, prediction) in enumerate(extremes):
            truth = np.zeros_like(image); valid = label != 255; truth[valid] = palette[label[valid]]
            predicted = palette[prediction]
            overlay = (image.astype(np.float32) * .5 + predicted.astype(np.float32) * .5).astype(np.uint8)
            y = i * (size + 24); draw.text((2, y + 4), f'{ident}  IoU {score:.3f}', fill='white')
            for j, panel in enumerate((image, truth, predicted, overlay)): canvas.paste(Image.fromarray(panel), (size * j, y + 24))
        args.qualitative.parent.mkdir(parents=True, exist_ok=True); canvas.save(args.qualitative)
    result.update({'status': 'complete', 'freeze_sha256': sha256(args.freeze), 'checkpoint_sha256': frozen['checkpoint_sha256'],
                   'split': 'test_min3', 'total_images': len(expected), 'evaluated_images': processed,
                   'skipped': skipped, 'per_image': per_image, 'ignored_pixels': pixels - result['valid_pixels'],
                   'evaluated_pixels': pixels, 'ignored_fraction': (pixels - result['valid_pixels']) / pixels,
                   'device': args.device, 'elapsed_seconds_including_io': time.monotonic() - start})
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n'); temporary.replace(args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
