import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from marsterrain.models import make_model
from marsterrain.train import PreparedDataset

parser = argparse.ArgumentParser()
parser.add_argument('--write-reference', type=Path)
args = parser.parse_args()
checkpoint_path = Path('checkpoints/mobile-scratch-v3/best.pt')
frozen = json.loads(Path('experiments/frozen-model.json').read_text())
if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != frozen['checkpoint_sha256']: raise ValueError('Checkpoint hash mismatch')
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
config = checkpoint['metadata']['config']
root = Path('data/processed/pairs256')
records = json.loads((root / 'prepared_manifest.json').read_text())['records']
record = next(r for r in records if r['id'] == 'NLB_443478972EDR_F0250886NCAM00258M1.JPG')
image, _ = PreparedDataset(root, [record], size=config['size'])[0]
model = make_model(config['architecture'], channels=config['channels'], pretrained=False).eval(); model.load_state_dict(checkpoint['model'])
with torch.no_grad(): expected = model(image.unsqueeze(0)).argmax(1)[0].numpy()
if args.write_reference:
    args.write_reference.write_bytes(expected.astype(np.uint8).tobytes())
    print(f'Wrote reference to {args.write_reference}')
    sys.exit(0)
browser = np.asarray(Image.open('data/processed/browser-mask.png').convert('RGB').resize((config['size'], config['size']), Image.Resampling.NEAREST))
palette = np.array([[196,127,73], [156,165,180], [231,197,116], [130,80,94]], dtype=np.uint8)
match = (browser[:, :, None, :] == palette[None, None, :, :]).all(axis=-1)
if not match.any(axis=-1).all(): raise ValueError('Unexpected browser mask colors')
actual = match.argmax(axis=-1)
print(json.dumps({'image_id': record['id'], 'argmax_agreement': float(np.mean(expected == actual)),
                  'scope': 'Python Pillow-resized image vs browser-decoded and resized original; 256px output'}))
