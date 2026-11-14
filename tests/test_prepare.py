import gzip
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location('prepare_data', Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_data.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def png(array):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(buffer, format='PNG')
    return buffer.getvalue()


class PreparationTest(unittest.TestCase):
    def test_prepare_and_refuse_expert_component(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = root / 'raw'; raw.mkdir()
            ident = 'NLA_123EDR_F0000001NCAM.JPG'
            image = png([[5, 7], [8, 9]])
            mask = png([[0, 1], [2, 255]])
            auxiliary = png([[0, 0], [1, 0]])
            struct = lambda path, contents: {'path': path, 'bytes': contents}
            row = {'image': struct(ident, image), 'label_mask': struct(ident[:-4] + '.png', mask),
                   'rover_mask': struct(ident[:-4].replace('EDR', 'MXY') + '.png', auxiliary),
                   'range_mask': struct(ident[:-4].replace('EDR', 'RNG') + '.png', png([[0, 0], [0, 0]]))}
            shard = 'train-00000-of-00013.parquet'
            pq.write_table(pa.Table.from_pylist([row]), raw / shard)
            entry = {'id': ident, 'shard': shard, 'row': 0, 'component': 'group-a', 'pixel_sha256': 'pixel',
                     'byte_sha256': hashlib.sha256(image).hexdigest(), 'assigned_split': 'train', 'source_split': 'train'}
            manifest = root / 'manifest.gz'
            def write(rows):
                with gzip.open(manifest, 'wt') as f:
                    for record in rows: f.write(json.dumps(record) + '\n')
            write([entry])
            module.prepare(raw, manifest, root / 'output', 4)
            result = json.loads((root / 'output' / 'prepared_manifest.json').read_text())
            self.assertEqual(len(result['records']), 1)
            mask_output = np.asarray(Image.open(root / 'output' / result['records'][0]['mask']))
            self.assertEqual(set(np.unique(mask_output)), {0, 1, 255})
            write([entry, {'id': 'expert', 'component': 'group-a', 'source_split': 'test_min3', 'assigned_split': 'test_min3'}])
            with self.assertRaisesRegex(ValueError, 'Expert'):
                module.prepare(raw, manifest, root / 'blocked', 4)

if __name__ == '__main__': unittest.main()
