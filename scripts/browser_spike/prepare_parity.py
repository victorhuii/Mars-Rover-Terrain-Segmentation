"""Write a real validation tensor and trusted PyTorch logits for browser parity."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from marsterrain.evaluate import sha256
from marsterrain.models import make_model
from marsterrain.train import PreparedDataset, sample_records

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--onnx', type=Path, required=True)
p.add_argument('--metadata', type=Path, required=True)
p.add_argument('--data', type=Path, default=Path('data/processed/pairs256'))
a = p.parse_args()
meta = json.loads(a.metadata.read_text())
if sha256(a.onnx) != meta['onnx_sha256'] or sha256(a.checkpoint) != meta['checkpoint_sha256']:
    raise ValueError('ONNX and checkpoint metadata hashes disagree')
c = torch.load(a.checkpoint, map_location='cpu', weights_only=False); cfg = c['metadata']['config']
manifest = a.data / 'prepared_manifest.json'
if sha256(manifest) != c['metadata']['prepared_manifest_sha256']: raise ValueError('Manifest mismatch')
records = sample_records(json.loads(manifest.read_text())['records'], 'val', 1, cfg['seed'] + 1)
x, _ = PreparedDataset(a.data, records, size=cfg['size'])[0]; x = x.unsqueeze(0)
model = make_model(cfg.get('architecture', 'unet'), channels=cfg['channels'], pretrained=False).eval()
model.load_state_dict(c['model'])
with torch.no_grad(): logits = model(x).numpy()
out = Path(__file__).resolve().parent
(out / 'input.bin').write_bytes(x.numpy().astype('<f4').tobytes())
(out / 'expected.bin').write_bytes(logits.astype('<f4').tobytes())
shutil.copyfile(a.onnx, out / 'model.onnx')
print(f'Prepared validation record {records[0]["id"]} for browser parity')
