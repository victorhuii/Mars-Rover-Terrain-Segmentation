import gzip
import hashlib
import json
from pathlib import Path
import unittest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class SampleProvenanceTest(unittest.TestCase):
    def test_samples_are_attributed_audited_training_images(self):
        directory = ROOT / 'frontend/public/samples'
        samples = json.loads((directory / 'samples.json').read_text())
        with gzip.open(ROOT / 'data/manifests/image_manifest.jsonl.gz', 'rt') as stream:
            audited = {row['id']: row for line in stream if (row := json.loads(line))['assigned_split'] == 'train'}
        self.assertGreaterEqual(len(samples), 4)
        self.assertEqual(len({row['sourceId'] for row in samples}), len(samples))
        for sample in samples:
            record = audited[sample['sourceId']]
            path = directory / Path(sample['file']).name
            self.assertFalse(record['expert_related'])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record['byte_sha256'])
            self.assertEqual(sample['sha256'], record['byte_sha256'])
            self.assertIn('NASA/JPL-Caltech', sample['credit'])
            with Image.open(path) as image: image.verify()

if __name__ == '__main__': unittest.main()
