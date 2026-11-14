import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

spec = importlib.util.spec_from_file_location('freeze_model', Path(__file__).resolve().parents[1] / 'scripts' / 'freeze_model.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class FreezeTest(unittest.TestCase):
    def test_freeze_binds_full_validation_and_scratch_lineage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifests = root / 'manifests'; manifests.mkdir()
            data = json.dumps({'records': [{'id': 'train', 'split': 'train'}, {'id': 'val', 'split': 'val'}]}).encode()
            with gzip.open(manifests / 'prepared_manifest.json.gz', 'wb') as f: f.write(data)
            audit = manifests / 'image_manifest.jsonl.gz'; audit.write_bytes(b'audit fixture')
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            (manifests / 'prepared_summary.json').write_text(json.dumps({'samples': {'val': 1}}))
            (manifests / 'audit_summary.json').write_text(json.dumps({'revision': 'fixture', 'manifest_sha256': digest(audit),
                'files': {'test_min3-00000-of-00001.parquet': {'sha256': 'expert-hash'}}}))
            checkpoint = root / 'model.pt'
            meta = {'config': {'architecture': 'unet', 'channels': 2, 'pretrained': False},
                    'overfit_validation_is_training': False, 'prepared_manifest_sha256': hashlib.sha256(data).hexdigest(),
                    'train_ids': ['train'], 'val_ids': ['val']}
            torch.save({'metadata': meta}, checkpoint)
            validation = root / 'validation.json'; validation.write_text(json.dumps({'checkpoint_sha256': digest(checkpoint),
                'split': 'validation', 'samples': 1, 'mean_iou': .5, 'per_class_iou': [.5, None, None, None]}))
            output = root / 'freeze.json'; args = ['freeze_model', '--checkpoint', str(checkpoint), '--validation', str(validation),
                                                  '--manifests', str(manifests), '--output', str(output)]
            with patch.object(sys, 'argv', args):
                module.main()
                with self.assertRaises(FileExistsError): module.main()
            self.assertEqual(json.loads(output.read_text())['checkpoint_sha256'], digest(checkpoint))
            meta['config']['pretrained'] = True; torch.save({'metadata': meta}, checkpoint)
            output.unlink()
            with patch.object(sys, 'argv', args):
                with self.assertRaisesRegex(ValueError, 'scratch-trained'): module.main()

if __name__ == '__main__': unittest.main()
