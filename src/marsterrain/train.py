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
from .models import make_model
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
    parser.add_argument('--architecture', choices=('unet', 'mobilenet_v3_small'), default='unet')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--class-weights', choices=('none', 'inverse-sqrt'), default='none')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--limit-train', type=int, default=0)
    parser.add_argument('--limit-val', type=int, default=0)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps'), default='auto')
    parser.add_argument('--resume', type=Path, help='Resume a trusted local last.pt checkpoint')
    parser.add_argument('--init-checkpoint', type=Path, help='Initialize a new run from a trusted scratch-trained checkpoint')
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.size < 16 or args.patience < 1 or
            args.channels < 1 or args.lr <= 0 or args.limit_train < 0 or args.limit_val < 0):
        parser.error('Invalid training parameters: sizes, counts and learning rate must be positive')
    if args.resume and args.init_checkpoint:
        parser.error('Choose resume or initialization, not both')
    if args.init_checkpoint and args.pretrained:
        parser.error('Initialization and pretrained weights are mutually exclusive')
    if args.pretrained and args.architecture != 'mobilenet_v3_small':
        parser.error('pretrained weights require mobilenet_v3_small')
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
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise ValueError('Output directory is not empty; use a new directory or resume last.pt')
    if args.resume and args.resume.resolve().parent != args.output.resolve():
        raise ValueError('Resume checkpoint must be in the output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy(); config['data'] = str(args.data); config['output'] = str(args.output); config['resume'] = str(args.resume) if args.resume else None; config['init_checkpoint'] = str(args.init_checkpoint) if args.init_checkpoint else None
    metadata = {'config': config, 'device': str(device), 'torch_version': torch.__version__,
                'prepared_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                'train_ids': [r['id'] for r in train], 'val_ids': [r['id'] for r in val],
                'overfit_validation_is_training': args.overfit}
    train_loader = DataLoader(PreparedDataset(args.data, train, not args.overfit, args.size), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(PreparedDataset(args.data, val, False, args.size), batch_size=args.batch_size, shuffle=False, num_workers=0)
    weights = None
    if args.class_weights == 'inverse-sqrt':
        counts = np.array([sum(int(r['pixel_counts'].get(str(i), 0)) for r in train) for i in range(4)], dtype=np.float64)
        if np.any(counts == 0): raise ValueError('Selected training subset lacks a class required for class weighting')
        weights_array = np.sqrt(counts.sum() / counts)
        weights_array /= weights_array.mean()
        weights = torch.tensor(weights_array, dtype=torch.float32, device=device)
        metadata['class_weights'] = weights_array.tolist()
    model = make_model(args.architecture, channels=args.channels, pretrained=args.pretrained).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
        previous = initial['metadata']
        previous_config = previous['config']
        if (previous_config.get('pretrained') or previous.get('overfit_validation_is_training') or
                previous_config.get('architecture', 'unet') != args.architecture or
                previous_config['channels'] != args.channels or previous['prepared_manifest_sha256'] != metadata['prepared_manifest_sha256']):
            raise ValueError('Initialization requires a compatible scratch-trained checkpoint and dataset')
        model.load_state_dict(initial['model'])
        metadata['initial_checkpoint_sha256'] = hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    best, stale, history, first_epoch, elapsed_before = -1.0, 0, [], 1, 0.0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        old = saved['metadata']
        old['config'].setdefault('init_checkpoint', None)
        ignored = {'output', 'epochs', 'resume', 'init_checkpoint'}
        if ({k: v for k, v in old['config'].items() if k not in ignored} !=
                {k: v for k, v in metadata['config'].items() if k not in ignored} or
                old['prepared_manifest_sha256'] != metadata['prepared_manifest_sha256'] or
                old['train_ids'] != metadata['train_ids'] or old['val_ids'] != metadata['val_ids']):
            raise ValueError('Resume configuration or data differ from the checkpoint')
        metadata['config']['init_checkpoint'] = old['config'].get('init_checkpoint')
        if 'initial_checkpoint_sha256' in old: metadata['initial_checkpoint_sha256'] = old['initial_checkpoint_sha256']
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        best, stale, history, first_epoch = saved['best'], saved['stale'], saved['history'], saved['epoch'] + 1
        elapsed_before = history[-1]['seconds_elapsed'] if history else 0.0
        random.setstate(saved['random_state']); np.random.set_state(saved['numpy_state']); torch.set_rng_state(saved['torch_state'])
        if device.type == 'mps' and saved.get('mps_state') is not None: torch.mps.set_rng_state(saved['mps_state'])
        if stale >= args.patience or first_epoch > args.epochs:
            print('Checkpoint has already reached the configured stopping criterion; no training resumed.')
            return
    (args.output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    start = time.monotonic()
    for epoch in range(first_epoch, args.epochs + 1):
        model.train(); train_loss = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(images), labels, weights)
            if not torch.isfinite(loss): raise ValueError('Non-finite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_loss.append(float(loss.item()))
        metrics = evaluate(model, val_loader, device)
        score = metrics['mean_iou']
        if score is None: raise ValueError('Validation has no valid pixels')
        if not args.overfit:
            scheduler.step(score)
        entry = {'epoch': epoch, 'train_loss': float(np.mean(train_loss)), 'validation': metrics,
                 'seconds_elapsed': elapsed_before + time.monotonic() - start, 'learning_rate': optimizer.param_groups[0]['lr']}
        history.append(entry)
        (args.output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        print(json.dumps(entry), flush=True)
        if score > best:
            best, stale = score, 0
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'score': score, 'metadata': metadata}, args.output / 'best.pt')
        else:
            stale += 1
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                 'epoch': epoch, 'best': best, 'stale': stale, 'history': history, 'metadata': metadata,
                 'random_state': random.getstate(), 'numpy_state': np.random.get_state(), 'torch_state': torch.get_rng_state(),
                 'mps_state': torch.mps.get_rng_state() if device.type == 'mps' else None}
        temporary = args.output / 'last.pt.tmp'
        torch.save(state, temporary); temporary.replace(args.output / 'last.pt')
        if stale >= args.patience: break


if __name__ == '__main__':
    main()
