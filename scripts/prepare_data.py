import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from marsterrain.dataset import decode_image, prepare_pair


def prepare(raw, manifest, output, size):
    with gzip.open(manifest, 'rt') as f:
        records = [json.loads(line) for line in f]
    selected = {(r['shard'], r['row']): r for r in records if r['assigned_split'] in ('train', 'val')}
    expert_ids = {r['id'] for r in records if r['source_split'].startswith('test_')}
    expert_components = {r['component'] for r in records if r['source_split'].startswith('test_')}
    if any(r['id'] in expert_ids or r['component'] in expert_components for r in selected.values()):
        raise ValueError('Expert identity or component present in training/validation manifest')
    output.mkdir(parents=True, exist_ok=True)
    entries, skipped = [], collections.Counter()
    seen = set()
    for shard in sorted({key[0] for key in selected}):
        path = raw / shard
        parquet = pq.ParquetFile(path)
        offset = 0
        for batch in parquet.iter_batches(batch_size=8):
            for record in batch.to_pylist():
                meta = selected.get((shard, offset))
                offset += 1
                if meta is None:
                    continue
                expected_stem = Path(meta['id']).stem
                expected_paths = {'label_mask': expected_stem, 'rover_mask': expected_stem.replace('EDR', 'MXY'), 'range_mask': expected_stem.replace('EDR', 'RNG')}
                blob = record['image']['bytes']
                if hashlib.sha256(blob).hexdigest() != meta['byte_sha256'] or record['image']['path'] != meta['id']:
                    raise ValueError(f'Shard identity mismatch for {shard}:{offset - 1}')
                if meta['pixel_sha256'] in seen:
                    skipped['duplicate_image'] += 1
                    continue
                if any(not record.get(k) or not record[k].get('bytes') for k in ('label_mask', 'rover_mask', 'range_mask')):
                    skipped['missing_mask'] += 1
                    continue
                if any(Path(record[k].get('path') or '').stem != stem for k, stem in expected_paths.items()):
                    raise ValueError(f'Image/mask filename mismatch for {shard}:{offset - 1}')
                try:
                    image, label = prepare_pair(*(decode_image(record[k]['bytes']) for k in ('image', 'label_mask', 'rover_mask', 'range_mask')), size=size)
                except (ValueError, OSError) as error:
                    skipped[str(error)] += 1
                    continue
                seen.add(meta['pixel_sha256'])
                key = f'{Path(shard).stem}-{offset - 1:05d}'
                dest = output / meta['assigned_split']
                dest.mkdir(exist_ok=True)
                image.save(dest / f'{key}.png')
                label.save(dest / f'{key}.mask.png')
                values, counts = np.unique(np.asarray(label), return_counts=True)
                entries.append({'id': meta['id'], 'component': meta['component'], 'split': meta['assigned_split'],
                                'image': f'{meta["assigned_split"]}/{key}.png', 'mask': f'{meta["assigned_split"]}/{key}.mask.png',
                                'pixel_counts': {str(int(k)): int(v) for k, v in zip(values, counts)}})
        print(f'{shard}: processed', flush=True)
    (output / 'prepared_manifest.json').write_text(json.dumps({'size': size, 'source_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                                                                'skipped': dict(skipped), 'records': entries}, indent=2) + '\n')
    print(f'Prepared {len(entries)} examples; skipped {dict(skipped)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=Path('data/raw'))
    parser.add_argument('--manifest', type=Path, default=Path('data/manifests/image_manifest.jsonl.gz'))
    parser.add_argument('--output', type=Path, default=Path('data/processed/pairs256'))
    parser.add_argument('--size', type=int, default=256)
    args = parser.parse_args()
    prepare(args.raw, args.manifest, args.output, args.size)
