import argparse
import json
from pathlib import Path
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from marsterrain.models import make_model

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--initial', type=Path, required=True)
a = p.parse_args()
import hashlib
initial_hash = hashlib.sha256(a.initial.read_bytes()).hexdigest()
selected = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
initial = torch.load(a.initial, map_location='cpu', weights_only=False)
if selected['metadata'].get('initial_checkpoint_sha256') != initial_hash:
    raise ValueError('Initialization checkpoint hash mismatch')
after, before = selected['model'], initial['model']
if after.keys() != before.keys(): raise ValueError('Checkpoint architectures differ')
config = selected['metadata']['config']
model = make_model(config.get('architecture', 'unet'), channels=config['channels'], pretrained=False)
parameters = {name for name, _ in model.named_parameters()}
changed, total, maximum = 0, 0, 0.0
for name, tensor in after.items():
    if name not in parameters or not tensor.is_floating_point(): continue
    delta = (tensor - before[name]).abs()
    total += 1
    if torch.any(delta != 0): changed += 1; maximum = max(maximum, float(delta.max()))
if not changed: raise ValueError('No floating tensors changed')
print(json.dumps({'initial_sha256': initial_hash, 'parameter_tensors': total,
                  'changed_parameter_tensors': changed, 'max_absolute_weight_change': maximum}, indent=2))
