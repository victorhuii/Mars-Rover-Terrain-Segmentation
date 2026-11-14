"""Independently check split identities, scene keys, hashes and perceptual links."""
import argparse
import collections
import gzip
import json
from pathlib import Path


def verify(path):
    with gzip.open(path, 'rt') as file:
        records = [json.loads(line) for line in file]
    if len({(r['source_split'], r['shard'], r['row']) for r in records}) != len(records):
        raise ValueError('Duplicate source rows')
    buckets = collections.defaultdict(list)
    seen = {}
    counted = collections.Counter()
    for row in records:
        split = row['assigned_split']
        if row['source_split'].startswith('test_'):
            if row['has_labels'] is not None or split != row['source_split']:
                raise ValueError('Expert metadata or split is incorrect')
            side = 'test'
        elif split in ('train', 'val'):
            if row['expert_related']:
                raise ValueError('Expert-related image assigned to training or validation')
            side = split
        else:
            continue
        counted[side] += 1
        for key in ('component', 'scene', 'id', 'byte_sha256', 'pixel_sha256'):
            value = row[key]
            if value is None:
                continue
            identity = (key, value)
            if identity in seen and seen[identity] != side:
                raise ValueError(f'{key} crosses splits')
            seen[identity] = side
        if row['dhash'] is None:
            continue
        value = int(row['dhash'], 16)
        candidates = set()
        for part in range(4):
            candidates.update(buckets[(part, (value >> (16 * part)) & 0xffff)])
        if any(other_side != side and (value ^ other_hash).bit_count() <= 2 for other_hash, other_side in candidates):
            raise ValueError('Near-duplicate image crosses splits')
        for part in range(4):
            buckets[(part, (value >> (16 * part)) & 0xffff)].append((value, side))
    if not counted['train'] or not counted['val'] or not counted['test']:
        raise ValueError(f'Empty required split: {dict(counted)}')
    print(f'Validated split disjointness: {dict(counted)}; test count includes label variants')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', nargs='?', type=Path, default=Path('data/manifests/image_manifest.jsonl.gz'))
    verify(parser.parse_args().manifest)
