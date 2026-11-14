import gzip
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location('audit_dataset', Path(__file__).resolve().parents[1] / 'scripts' / 'audit_dataset.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AuditTest(unittest.TestCase):
    def test_expert_image_linked_without_reading_label_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = root / 'raw'; raw.mkdir(); output = root / 'manifest'
            buffer = io.BytesIO(); Image.fromarray(np.arange(256, dtype=np.uint8).reshape(16, 16)).save(buffer, format='PNG')
            image = {'path': 'NLA_123EDR_F0000001NCAM.JPG', 'bytes': buffer.getvalue()}
            train_name, test_name = 'train-00000-of-00013.parquet', 'test_min3-00000-of-00001.parquet'
            pq.write_table(pa.Table.from_pylist([{'image': image, 'has_labels': True}]), raw / train_name)
            pq.write_table(pa.Table.from_pylist([{'image': image, 'label_mask': 'should not be read'}]), raw / test_name)
            original = module.FILES; module.FILES = [train_name, test_name]
            try:
                module.audit(raw, output)
                with gzip.open(output / 'image_manifest.jsonl.gz', 'rt') as f:
                    rows = [json.loads(line) for line in f]
                train = next(x for x in rows if x['source_split'] == 'train')
                test = next(x for x in rows if x['source_split'] == 'test_min3')
                self.assertEqual(train['assigned_split'], 'excluded_expert_related')
                self.assertIsNone(test['has_labels'])
                self.assertNotIn('label_mask', test)
            finally:
                module.FILES = original

if __name__ == '__main__': unittest.main()
