import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from marsterrain.evaluate import sha256
from marsterrain.models import make_model
from marsterrain.train import PreparedDataset, sample_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/processed/pairs256'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--iterations', type=int, default=30)
    args = parser.parse_args()
    if args.iterations < 1: parser.error('iterations must be positive')
    if args.device == 'mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    manifest = args.data / 'prepared_manifest.json'
    if sha256(manifest) != checkpoint['metadata']['prepared_manifest_sha256']: raise ValueError('Manifest mismatch')
    config = checkpoint['metadata']['config']; device = torch.device(args.device)
    records = sample_records(json.loads(manifest.read_text())['records'], 'val', 1, config['seed'] + 1)
    image, _ = PreparedDataset(args.data, records, size=config['size'])[0]
    x = image.unsqueeze(0).to(device)
    model = make_model(config.get('architecture', 'unet'), channels=config['channels'], pretrained=False).to(device)
    model.load_state_dict(checkpoint['model']); model.eval()
    def sync():
        if device.type == 'mps': torch.mps.synchronize()
    samples = []
    with torch.no_grad():
        for iteration in range(args.iterations + 5):
            sync(); start = time.perf_counter(); model(x); sync()
            if iteration >= 5: samples.append((time.perf_counter() - start) * 1000)
    result = {'checkpoint_sha256': sha256(args.checkpoint), 'device': args.device, 'image_id': records[0]['id'],
              'batch_size': 1, 'input_size': config['size'], 'warmup': 5, 'iterations': args.iterations,
              'cpu_threads': torch.get_num_threads(), 'median_ms': statistics.median(samples), 'min_ms': min(samples), 'max_ms': max(samples),
              'scope': 'eager PyTorch forward only; image decoding, model loading and browser runtime excluded'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n'); print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
