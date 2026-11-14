import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image


class TrainingCliTest(unittest.TestCase):
    def test_train_and_validation_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = root / 'data'; data.mkdir()
            records = []
            for split in ('train', 'val'):
                image = f'{split}.png'; mask = f'{split}.mask.png'
                Image.new('RGB', (16, 16), (100, 100, 100)).save(data / image)
                Image.new('L', (16, 16), 0).save(data / mask)
                records.append({'id': split, 'split': split, 'image': image, 'mask': mask})
            (data / 'prepared_manifest.json').write_text(json.dumps({'records': records}))
            env = os.environ.copy(); env['OMP_NUM_THREADS'] = '1'; env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
            checkpoint = root / 'run' / 'best.pt'
            subprocess.run([sys.executable, '-m', 'marsterrain.train', '--data', str(data), '--output', str(checkpoint.parent),
                            '--epochs', '1', '--patience', '1', '--size', '16', '--channels', '2', '--batch-size', '1', '--device', 'cpu'],
                           env=env, check=True, capture_output=True, text=True, timeout=45)
            output = root / 'validation.json'
            subprocess.run([sys.executable, '-m', 'marsterrain.evaluate', '--data', str(data), '--checkpoint', str(checkpoint), '--output', str(output)],
                           env=env, check=True, capture_output=True, text=True, timeout=45)
            result = json.loads(output.read_text())
            self.assertEqual(result['samples'], 1)
            self.assertEqual(result['valid_pixels'], 256)
            self.assertEqual(result['split'], 'validation')
            self.assertEqual(len(result['checkpoint_sha256']), 64)

if __name__ == '__main__': unittest.main()
