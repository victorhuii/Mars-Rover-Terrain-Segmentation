"""Audit all pinned image columns, without reading any expert label columns."""
import argparse
import collections
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from marsterrain.splits import dhash, group_records, scene_key

REVISION = '93dbb79de6cd234799750b6129fbfb3b197582fa'
FILES = ([f'train-{i:05d}-of-00013.parquet' for i in range(13)] +
         [f'test_min{i}-00000-of-00001.parquet' for i in range(1, 4)])


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def audit(raw, output):
    missing = [f for f in FILES if not (raw / f).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing pinned shards: {missing}')
    records, shards = [], {}
    cache = raw.parent / 'cache' / 'image-audit-v1'
    cache.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = raw / name
        split = name.split('-')[0]
        parquet = pq.ParquetFile(path)
        shards[name] = {'sha256': file_hash(path), 'rows': parquet.metadata.num_rows}
        cached = cache / f'{name}.{shards[name]["sha256"]}.json'
        if cached.exists():
            shard_records = json.loads(cached.read_text())
            if len(shard_records) != parquet.metadata.num_rows:
                raise ValueError(f'Invalid audit cache: {cached}')
            records.extend(shard_records)
            print(f'{name}: {len(shard_records)} cached images audited', flush=True)
            continue
        shard_records = []
        # Never request an expert label, mask, or derived label property.
        columns = ['image', 'has_labels'] if split == 'train' else ['image']
        offset = 0
        for batch in parquet.iter_batches(batch_size=16, columns=columns):
            for item in batch.to_pylist():
                image = item['image']
                if not image or not image.get('path') or not image.get('bytes'):
                    raise ValueError(f'Missing image identity/data: {name} row {offset}')
                content = image['bytes']
                try:
                    with Image.open(io.BytesIO(content)) as img:
                        img.load()
                        rgb = img.convert('RGB')
                        pixels = hashlib.sha256(rgb.size[0].to_bytes(4, 'big') + rgb.size[1].to_bytes(4, 'big') + rgb.tobytes()).hexdigest()
                        perceptual = f'{dhash(img):016x}'
                        size = list(img.size)
                    error = None
                except (OSError, ValueError) as exc:
                    pixels, perceptual, size, error = None, None, None, type(exc).__name__
                try:
                    scene = scene_key(image['path'])
                except ValueError:
                    if split != 'train':
                        raise ValueError(f'Unrecognized expert image identity: {image["path"]}')
                    scene, error = f'unknown:{image["path"]}', error or 'UnrecognizedProductName'
                shard_records.append({'source_split': split, 'shard': name, 'row': offset,
                                'id': image['path'], 'scene': scene,
                                'byte_sha256': hashlib.sha256(content).hexdigest(), 'pixel_sha256': pixels,
                                'dhash': perceptual, 'size': size, 'image_error': error,
                                'has_labels': item.get('has_labels') if split == 'train' else None})
                offset += 1
        tmp = cached.with_suffix('.tmp')
        tmp.write_text(json.dumps(shard_records, separators=(',', ':')))
        tmp.replace(cached)
        records.extend(shard_records)
        print(f'{name}: {offset} images audited', flush=True)
    group_stats = group_records(records)
    records.sort(key=lambda r: (r['source_split'], r['shard'], r['row']))
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'image_manifest.jsonl.gz'
    with manifest.open('wb') as raw_output:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw_output, mtime=0) as compressed:
            for row in records:
                compressed.write((json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n').encode())
    counts = collections.Counter(row['assigned_split'] for row in records)
    summary = {'revision': REVISION, 'seed': 42, 'validation_fraction_target': 0.15,
               'near_duplicate_dhash_max_distance': 2, 'files': shards,
               'assigned_counts': dict(sorted(counts.items())), 'grouping': group_stats,
               'manifest_sha256': file_hash(manifest), 'expert_labels_read': False}
    (output / 'audit_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=Path('data/raw'))
    parser.add_argument('--output', type=Path, default=Path('data/manifests'))
    args = parser.parse_args()
    audit(args.raw, args.output)
