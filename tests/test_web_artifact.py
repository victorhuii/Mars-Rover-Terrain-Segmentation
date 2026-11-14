"""Integrity and real-inference smoke test for the frozen deployment artifact."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'frontend/public/models/marsterrain-v1-be2513a68f69.onnx'
METADATA = MODEL.with_suffix('.json')


class WebArtifactTest(unittest.TestCase):
    def test_hash_lineage_and_real_input_dependent_logits(self):
        metadata = json.loads(METADATA.read_text())
        frozen = json.loads((ROOT / 'experiments/frozen-model.json').read_text())
        expert = json.loads((ROOT / 'results/expert-test-min3.json').read_text())
        self.assertEqual(hashlib.sha256(MODEL.read_bytes()).hexdigest(), metadata['onnx_sha256'])
        self.assertEqual(metadata['checkpoint_sha256'], frozen['checkpoint_sha256'])
        self.assertEqual(metadata['checkpoint_sha256'], expert['checkpoint_sha256'])
        self.assertLessEqual(MODEL.stat().st_size, 25 * 1024 * 1024)
        options = ort.SessionOptions(); options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(MODEL), sess_options=options, providers=['CPUExecutionProvider'])
        low = session.run(None, {'image': np.zeros((1, 3, 256, 256), dtype=np.float32)})[0]
        high = session.run(None, {'image': np.ones((1, 3, 256, 256), dtype=np.float32)})[0]
        self.assertEqual(low.shape, (1, 4, 256, 256))
        self.assertTrue(np.isfinite(low).all() and np.isfinite(high).all())
        self.assertGreater(float(np.max(np.abs(low - high))), 1e-5)

if __name__ == '__main__': unittest.main()
