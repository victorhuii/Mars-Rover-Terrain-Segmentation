"""Conservative image identity grouping for leakage-safe AI4Mars splits.

No expert annotations are needed to use this module. Site/drive grouping uses
an identifier in the product name as a conservative proxy, not a guarantee of
geographic independence. Similarity hashing is a screening heuristic, not a
proof that all related scenes have been identified.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict

SCENE = re.compile(r'_F(\d{7})', re.IGNORECASE)


def scene_key(name: str) -> str:
    match = SCENE.search(name)
    if not match:
        raise ValueError(f'Unrecognized Curiosity product name: {name}')
    return 'F' + match.group(1)


def dhash(image) -> int:
    """64-bit horizontal difference hash of the decoded image."""
    from PIL import Image
    pixels = list(image.convert('L').resize((9, 8), Image.Resampling.LANCZOS).getdata())
    result = 0
    for y in range(8):
        for x in range(8):
            result = (result << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return result


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def group_records(records, seed=42, near_distance=2):
    """Link scene groups, exact IDs/hashes and near images, then split groups.

    Four 16-bit hash buckets find all pairs with Hamming distance <= 2.
    Any component containing an expert image is entirely held out from training.
    Splits depend only on seed and component members, not record order.
    """
    if not 0 <= near_distance <= 3:
        raise ValueError('near_distance must be between 0 and 3')
    uf = UnionFind(len(records))
    seen = {}
    buckets = defaultdict(list)
    near_pairs = 0
    for i, row in enumerate(records):
        for field in ('scene', 'id', 'byte_sha256', 'pixel_sha256'):
            if row[field] is None:
                continue
            key = (field, row[field])
            if key in seen:
                uf.union(i, seen[key])
            else:
                seen[key] = i
        if row['dhash'] is None:
            continue
        value = int(row['dhash'], 16)
        candidates = set()
        for part in range(4):
            candidates.update(buckets[(part, (value >> (part * 16)) & 0xffff)])
        for j in candidates:
            if (value ^ int(records[j]['dhash'], 16)).bit_count() <= near_distance:
                uf.union(i, j)
                near_pairs += 1
        for part in range(4):
            buckets[(part, (value >> (part * 16)) & 0xffff)].append(i)
    components = defaultdict(list)
    for i, row in enumerate(records):
        components[uf.find(i)].append(row)
    for members in components.values():
        component_id = hashlib.sha256('\n'.join(sorted({r['id'] for r in members})).encode()).hexdigest()
        expert = any(r['source_split'].startswith('test_') for r in members)
        fraction = int(hashlib.sha256(f'{seed}:{component_id}'.encode()).hexdigest()[:16], 16) / 2**64
        split = 'excluded_expert_related' if expert else ('val' if fraction < 0.15 else 'train')
        for row in members:
            row['component'] = component_id
            row['assigned_split'] = (row['source_split'] if row['source_split'].startswith('test_') else
                                     'excluded_corrupt' if row.get('image_error') else
                                     split if row['has_labels'] else 'excluded_unlabelled')
            row['expert_related'] = expert
    return {'components': len(components), 'near_pairs': near_pairs}
