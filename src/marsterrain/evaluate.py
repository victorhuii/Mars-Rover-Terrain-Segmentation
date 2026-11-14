import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .models import make_model
from .train import PreparedDataset, evaluate, sample_records


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/processed/pairs256'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    args = parser.parse_args()
    if args.batch_size <= 0 or args.limit < 0: parser.error('Invalid batch size or limit')
    if args.device == 'mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS is unavailable')

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['metadata']['overfit_validation_is_training']:
        raise ValueError('Overfit diagnostics cannot be presented as validation')
    config = checkpoint['metadata']['config']
    manifest_path = args.data / 'prepared_manifest.json'
    if sha256(manifest_path) != checkpoint['metadata']['prepared_manifest_sha256']:
        raise ValueError('Prepared manifest differs from training')
    prepared = json.loads(manifest_path.read_text())
    records = sample_records(prepared['records'], 'val', args.limit, config['seed'] + 1)
    if not records: raise ValueError('Empty validation split')
    device = torch.device(args.device)
    model = make_model(config.get('architecture', 'unet'), channels=config['channels'], pretrained=False).to(device)
    model.load_state_dict(checkpoint['model'])
    loader = DataLoader(PreparedDataset(args.data, records, False, config['size']), batch_size=args.batch_size, num_workers=0)
    start = time.monotonic(); result = evaluate(model, loader, device); elapsed = time.monotonic() - start
    result.update({'split': 'validation', 'samples': len(records), 'seed': config['seed'],
                   'checkpoint_sha256': sha256(args.checkpoint), 'checkpoint_bytes': args.checkpoint.stat().st_size,
                   'checkpoint_epoch': checkpoint['epoch'], 'device': str(device), 'elapsed_seconds_including_io': elapsed})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
