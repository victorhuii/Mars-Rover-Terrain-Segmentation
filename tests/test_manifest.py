import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('verify_manifest', Path(__file__).resolve().parents[1] / 'scripts' / 'verify_manifest.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(name, source, assigned, scene, h):
    return {'source_split': source, 'assigned_split': assigned, 'has_labels': None if source.startswith('test_') else True,
            'shard': source, 'row': 0, 'id': name, 'component': name, 'scene': scene, 'expert_related': source.startswith('test_'),
            'byte_sha256': name, 'pixel_sha256': name, 'dhash': f'{h:016x}'}


class ManifestTest(unittest.TestCase):
    def test_disjointness_and_near_duplicate_detection(self):
        rows = [row('a', 'train', 'train', 'a', 0), row('b', 'train', 'val', 'b', 0xaaaaaaaaaaaaaaaa),
                row('c', 'test_min3', 'test_min3', 'c', 0xffffffffffffffff)]
        # Rows from a common shard must have distinct row numbers.
        rows[1]['row'] = 1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.gz'
            def write():
                with gzip.open(path, 'wt') as output:
                    for entry in rows: output.write(json.dumps(entry) + '\n')
            write(); module.verify(path)
            rows[1]['scene'] = 'a'; write()
            with self.assertRaisesRegex(ValueError, 'scene crosses'): module.verify(path)
            rows[1]['scene'] = 'b'; rows[1]['dhash'] = '0000000000000001'; write()
            with self.assertRaisesRegex(ValueError, 'Near-duplicate'): module.verify(path)

if __name__ == '__main__': unittest.main()
