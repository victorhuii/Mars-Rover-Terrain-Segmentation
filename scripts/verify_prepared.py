import collections
import gzip
import hashlib
import json
from pathlib import Path

raw_manifest = Path('data/manifests/image_manifest.jsonl.gz')
root = Path('data/processed/pairs256')
prepared = json.loads((root / 'prepared_manifest.json').read_text())
if hashlib.sha256(raw_manifest.read_bytes()).hexdigest() != prepared['source_manifest_sha256']:
    raise ValueError('Source manifest hash differs')
with gzip.open(raw_manifest, 'rt') as f:
    source = {r['id']: r for line in f if (r := json.loads(line))['assigned_split'] in ('train', 'val')}
counts, pixels, seen = collections.Counter(), collections.defaultdict(collections.Counter), set()
for record in prepared['records']:
    ident = record['id']
    if ident in seen or ident not in source:
        raise ValueError(f'Duplicate or missing source: {ident}')
    seen.add(ident)
    if record['split'] != source[ident]['assigned_split'] or record['component'] != source[ident]['component']:
        raise ValueError(f'Split or component mismatch: {ident}')
    if not (root / record['image']).is_file() or not (root / record['mask']).is_file():
        raise ValueError(f'Missing output: {ident}')
    counts[record['split']] += 1
    pixels[record['split']].update({k: int(v) for k, v in record['pixel_counts'].items()})
for split in ('train', 'val'):
    if not counts[split]: raise ValueError(f'Empty split: {split}')
print(json.dumps({'samples': dict(counts), 'pixel_counts': {k: dict(v) for k, v in pixels.items()},
                  'skipped': prepared['skipped']}, indent=2))
