"""Export four attributed, audited train images without labels for the demo."""
import gzip
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq

SAMPLES = [
    ('plain', 'Open plain', 'NLB_443478972EDR_F0250886NCAM00258M1.JPG'),
    ('bedrock', 'Fractured ground', 'NLB_533948978EDR_F0592830NCAM00323M1.JPG'),
    ('dunes', 'Wind-shaped sand', 'NLB_460612405EDR_F0400540NCAM00259M1.JPG'),
    ('outcrop', 'Rocky outcrop', 'NLB_490980444EDR_F0482518NCAM07753M1.JPG'),
]
with gzip.open('data/manifests/image_manifest.jsonl.gz', 'rt') as file:
    records = {r['id']: r for line in file if (r := json.loads(line))['assigned_split'] == 'train'}
out = Path('frontend/public/samples'); out.mkdir(parents=True, exist_ok=True)
items = []
for slug, title, ident in SAMPLES:
    info = records[ident]
    if info['expert_related']: raise ValueError('Expert-related sample')
    parquet = pq.ParquetFile(Path('data/raw') / info['shard'])
    offset = 0; found = None
    for batch in parquet.iter_batches(batch_size=16, columns=['image']):
        rows = batch.to_pylist()
        if offset <= info['row'] < offset + len(rows):
            found = rows[info['row'] - offset]['image']; break
        offset += len(rows)
    if not found or found['path'] != ident or hashlib.sha256(found['bytes']).hexdigest() != info['byte_sha256']:
        raise ValueError(f'Source mismatch: {ident}')
    (out / f'{slug}.jpg').write_bytes(found['bytes'])
    items.append({'id': slug, 'title': title, 'file': f'/samples/{slug}.jpg', 'sourceId': ident,
                  'sha256': info['byte_sha256'], 'credit': 'NASA/JPL-Caltech; AI4Mars, Swan et al.'})
(out / 'samples.json').write_text(json.dumps(items, indent=2) + '\n')
print(f'Exported {len(items)} attributed train images')
