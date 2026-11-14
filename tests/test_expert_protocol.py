import json
import gzip
import hashlib
import io
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import torch
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from marsterrain import evaluate_expert
from marsterrain.models import UNet


class ExpertProtocolTest(unittest.TestCase):
    def test_invalid_protocol_refuses_before_any_expert_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'freeze.json'
            path.write_text(json.dumps({'split': 'test_min3', 'protocol': 'unfrozen'}))
            args = ['evaluate_expert', '--freeze', str(path), '--checkpoint', str(Path(tmp) / 'missing'), '--output', str(Path(tmp) / 'result')]
            with patch.object(sys, 'argv', args), patch.object(evaluate_expert.pq, 'ParquetFile') as parquet:
                with self.assertRaisesRegex(ValueError, 'protocol'): evaluate_expert.main()
                parquet.assert_not_called()


    def test_mismatched_checkpoint_refuses_before_expert_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); freeze = root / 'freeze.json'; checkpoint = root / 'weights.pt'
            checkpoint.write_bytes(b'wrong checkpoint')
            freeze.write_text(json.dumps({'split': 'test_min3', 'protocol': 'ai4mars-expert-min3-v1', 'checkpoint_sha256': '0' * 64}))
            args = ['evaluate_expert', '--freeze', str(freeze), '--checkpoint', str(checkpoint), '--output', str(root / 'result')]
            with patch.object(sys, 'argv', args), patch.object(evaluate_expert.pq, 'ParquetFile') as parquet:
                with self.assertRaisesRegex(ValueError, 'checkpoint_sha256'): evaluate_expert.main()
                parquet.assert_not_called()


    def test_frozen_synthetic_evaluation_is_single_use(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = root / 'raw'; raw.mkdir(); manifests = root / 'manifests'; manifests.mkdir()
            def encode(array):
                file = io.BytesIO(); Image.fromarray(np.array(array, dtype=np.uint8)).save(file, format='PNG'); return file.getvalue()
            image = encode(np.full((16, 16), 80)); mask = encode(np.zeros((16, 16))); aux = encode(np.zeros((16, 16)))
            item = lambda name, data: {'path': name, 'bytes': data}
            rows = [{'image': item(f'img{i}', image), 'label_mask': item(f'label{i}', mask),
                     'rover_mask': item(f'rover{i}', aux), 'range_mask': item(f'range{i}', aux)} for i in range(2)]
            shard = raw / 'test_min3-00000-of-00001.parquet'; pq.write_table(pa.Table.from_pylist(rows), shard)
            audit = manifests / 'image_manifest.jsonl.gz'
            with gzip.open(audit, 'wt') as f:
                for i in range(2): f.write(json.dumps({'source_split': 'test_min3', 'row': i, 'id': f'img{i}', 'byte_sha256': hashlib.sha256(image).hexdigest()}) + '\n')
            prepared = manifests / 'prepared_manifest.json.gz'; prepared.write_bytes(b'fixture manifest')
            checkpoint = root / 'model.pt'
            torch.save({'model': UNet(channels=2).state_dict(), 'metadata': {'overfit_validation_is_training': False,
                       'prepared_manifest_sha256': 'fixture', 'config': {'channels': 2, 'size': 16}}}, checkpoint)
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            freeze = root / 'freeze.json'; freeze.write_text(json.dumps({'split': 'test_min3', 'protocol': 'ai4mars-expert-min3-v1',
                'checkpoint_sha256': digest(checkpoint), 'audit_manifest_sha256': digest(audit),
                'prepared_manifest_gzip_sha256': digest(prepared), 'expert_shard_sha256': digest(shard), 'prepared_manifest_sha256': 'fixture'}))
            result = root / 'expert.json'
            args = ['evaluate_expert', '--freeze', str(freeze), '--checkpoint', str(checkpoint), '--raw', str(raw),
                    '--manifests', str(manifests), '--output', str(result)]
            with patch.object(sys, 'argv', args), patch.object(evaluate_expert, 'EXPECTED_EXPERT_COUNT', 2):
                evaluate_expert.main()
                with self.assertRaises(FileExistsError): evaluate_expert.main()
            output = json.loads(result.read_text())
            self.assertEqual(output['evaluated_images'], 2)
            self.assertEqual(output['valid_pixels'], 512)
            self.assertEqual(output['status'], 'complete')

if __name__ == '__main__': unittest.main()
