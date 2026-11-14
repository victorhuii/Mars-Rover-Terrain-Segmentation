"""Seeded, resource-bounded U-Net training with validation-only selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from .losses import segmentation_loss
from .metrics import confusion_matrix, summarize
from .models import UNet
from .transforms import transform_pair


class PreparedDataset(Dataset):
    def __init__(self, root, records, augment=False, size=256):
        self.root, self.records, self.augment, self.size = root, records, augment, size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record['image']) as file:
            image = file.convert('RGB')
        with Image.open(self.root / record['mask']) as file:
            mask = file.copy()
        if image.size != (self.size, self.size):
            image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
            mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        if self.augment:
            image, mask = transform_pair(image, mask, flip=bool(torch.rand(()) < 0.5), brightness=float(0.9 + torch.rand(()) * 0.2))
        x = torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1).float().div_(255)
        y = torch.from_numpy(np.array(mask, dtype=np.int64))
        return x, y


def sample_records(records, split, limit, seed):
    candidates = [r for r in records if r['split'] == split]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:limit] if limit else candidates


def evaluate(model, loader, device):
    model.eval(); matrix = np.zeros((4, 4), dtype=np.int64); losses = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            losses.append(float(segmentation_loss(logits, labels).item()))
            matrix += confusion_matrix(labels.cpu().numpy(), logits.argmax(1).cpu().numpy())
    result = summarize(matrix)
    result['loss'] = float(np.mean(losses)) if losses else None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/processed/pairs256'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--limit-train', type=int, default=0)
    parser.add_argument('--limit-val', type=int, default=0)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps'), default='auto')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.size < 16 or args.patience < 1:
        parser.error('epochs, batch-size and patience must be positive; size must be at least 16')
    if args.overfit and not 1 <= args.limit_train <= 4:
        parser.error('overfit requires --limit-train between 1 and 4')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS was requested but is unavailable')
    device = torch.device('mps' if args.device == 'mps' or (args.device == 'auto' and torch.backends.mps.is_available()) else 'cpu')
    manifest_path = args.data / 'prepared_manifest.json'
    prepared = json.loads(manifest_path.read_text())
    train = sample_records(prepared['records'], 'train', args.limit_train, args.seed)
    val = train if args.overfit else sample_records(prepared['records'], 'val', args.limit_val, args.seed + 1)
    if not train or not val:
        raise ValueError('Training and validation subsets must not be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy(); config['data'] = str(args.data); config['output'] = str(args.output)
    metadata = {'config': config, 'device': str(device), 'torch_version': torch.__version__,
                'prepared_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                'train_ids': [r['id'] for r in train], 'val_ids': [r['id'] for r in val],
                'overfit_validation_is_training': args.overfit}
    (args.output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    train_loader = DataLoader(PreparedDataset(args.data, train, not args.overfit, args.size), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(PreparedDataset(args.data, val, False, args.size), batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = UNet(channels=args.channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    best, stale, history = -1.0, 0, []
    start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train(); train_loss = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(images), labels)
            if not torch.isfinite(loss): raise ValueError('Non-finite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_loss.append(float(loss.item()))
        metrics = evaluate(model, val_loader, device)
        score = metrics['mean_iou']
        if score is None: raise ValueError('Validation has no valid pixels')
        if not args.overfit:
            scheduler.step(score)
        entry = {'epoch': epoch, 'train_loss': float(np.mean(train_loss)), 'validation': metrics,
                 'seconds_elapsed': time.monotonic() - start, 'learning_rate': optimizer.param_groups[0]['lr']}
        history.append(entry)
        (args.output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        print(json.dumps(entry), flush=True)
        if score > best:
            best, stale = score, 0
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'score': score, 'metadata': metadata}, args.output / 'best.pt')
        else:
            stale += 1
            if stale >= args.patience: break


if __name__ == '__main__':
    main()
