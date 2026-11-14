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
            last = checkpoint.parent / 'last.pt'
            self.assertTrue(last.is_file())
            subprocess.run([sys.executable, '-m', 'marsterrain.train', '--data', str(data), '--output', str(checkpoint.parent),
                            '--epochs', '2', '--patience', '1', '--size', '16', '--channels', '2', '--batch-size', '1',
                            '--device', 'cpu', '--resume', str(last)], env=env, check=True, capture_output=True, text=True, timeout=45)
            self.assertEqual(len(json.loads((checkpoint.parent / 'history.json').read_text())), 2)
            initialized = root / 'initialized'
            command = [sys.executable, '-m', 'marsterrain.train', '--data', str(data), '--output', str(initialized),
                       '--epochs', '1', '--patience', '1', '--size', '16', '--channels', '2', '--batch-size', '1', '--device', 'cpu']
            subprocess.run(command + ['--init-checkpoint', str(checkpoint)], env=env, check=True, capture_output=True, text=True, timeout=45)
            subprocess.run([*command[:command.index('--epochs')], '--epochs', '2', *command[command.index('--patience'):],
                            '--resume', str(initialized / 'last.pt')], env=env, check=True, capture_output=True, text=True, timeout=45)
            metadata = json.loads((initialized / 'run.json').read_text())
            self.assertEqual(len(metadata['initial_checkpoint_sha256']), 64)
            output = root / 'validation.json'
            subprocess.run([sys.executable, '-m', 'marsterrain.evaluate', '--data', str(data), '--checkpoint', str(checkpoint), '--output', str(output)],
                           env=env, check=True, capture_output=True, text=True, timeout=45)
            export = root / 'model.onnx'; metadata_path = root / 'model.json'
            subprocess.run([sys.executable, '-m', 'marsterrain.export_onnx', '--data', str(data), '--checkpoint', str(checkpoint),
                            '--output', str(export), '--metadata', str(metadata_path), '--samples', '1'],
                           env=env, check=True, capture_output=True, text=True, timeout=90)
            export_meta = json.loads(metadata_path.read_text())
            self.assertEqual(export_meta['input_shape'], [1, 3, 16, 16])
            self.assertGreaterEqual(export_meta['parity'][0]['argmax_agreement'], 0.999)
            result = json.loads(output.read_text())
            self.assertEqual(result['samples'], 1)
            self.assertEqual(result['valid_pixels'], 256)
            self.assertEqual(result['split'], 'validation')
            self.assertEqual(len(result['checkpoint_sha256']), 64)

if __name__ == '__main__': unittest.main()
