"""Freeze a scratch-trained checkpoint and validation protocol before expert labels."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from marsterrain.evaluate import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--lineage', type=Path, nargs='*', default=[])
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, default=Path('data/manifests'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Frozen protocol already exists')
    paths = [args.checkpoint, *args.lineage]
    checkpoints = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
    for i, checkpoint in enumerate(checkpoints):
        meta = checkpoint['metadata']; config = meta['config']
        if config.get('pretrained') or meta['overfit_validation_is_training']:
            raise ValueError('Only non-overfit, scratch-trained lineage can be frozen')
        parent = meta.get('initial_checkpoint_sha256')
        if i + 1 < len(paths):
            if parent != sha256(paths[i + 1]): raise ValueError('Initialization lineage hash mismatch')
        elif parent:
            raise ValueError('Initialization lineage is incomplete')
    selected = checkpoints[0]; config = selected['metadata']['config']; selected_hash = sha256(args.checkpoint)
    val = json.loads(args.validation.read_text()); prepared_summary = json.loads((args.manifests / 'prepared_summary.json').read_text())
    if val['checkpoint_sha256'] != selected_hash or val['split'] != 'validation' or val['samples'] != prepared_summary['samples']['val']:
        raise ValueError('Full validation result does not match selected checkpoint')
    audit = args.manifests / 'image_manifest.jsonl.gz'; prepared = args.manifests / 'prepared_manifest.json.gz'
    summary = json.loads((args.manifests / 'audit_summary.json').read_text())
    if sha256(audit) != summary['manifest_sha256']: raise ValueError('Audit manifest mismatch')
    with gzip.open(prepared, 'rb') as file: prepared_bytes = file.read()
    prepared_hash = hashlib.sha256(prepared_bytes).hexdigest()
    split_by_id = {row['id']: row['split'] for row in json.loads(prepared_bytes)['records']}
    for checkpoint in checkpoints:
        meta = checkpoint['metadata']
        if meta['prepared_manifest_sha256'] != prepared_hash:
            raise ValueError('Training dataset manifest mismatch')
        if not meta['train_ids'] or not meta['val_ids'] or any(split_by_id.get(i) != 'train' for i in meta['train_ids']) or any(split_by_id.get(i) != 'val' for i in meta['val_ids']):
            raise ValueError('Training/validation identities do not match clean splits')
    frozen = {'protocol': 'ai4mars-expert-min3-v1', 'split': 'test_min3',
              'dataset_revision': summary['revision'], 'checkpoint_sha256': selected_hash,
              'initialization_lineage_sha256': [sha256(p) for p in args.lineage],
              'audit_manifest_sha256': sha256(audit), 'prepared_manifest_gzip_sha256': sha256(prepared),
              'prepared_manifest_sha256': selected['metadata']['prepared_manifest_sha256'],
              'expert_shard_sha256': summary['files']['test_min3-00000-of-00001.parquet']['sha256'],
              'validation_result_sha256': sha256(args.validation), 'validation_miou': val['mean_iou'],
              'validation_per_class_iou': val['per_class_iou'], 'model_config': config,
              'classes': ['soil', 'bedrock', 'sand', 'big rock'], 'ignore_index': 255,
              'palette_rgb': [[196,127,73], [156,165,180], [231,197,116], [130,80,94]],
              'preprocessing': ('RGB, bilinear resize to model size, divide by 255, model-internal ImageNet normalization' if config.get('architecture') == 'mobilenet_v3_small' else 'RGB, bilinear resize to model size, divide by 255'),
              'reference_masks': 'apply binary rover and range exclusions; nearest-neighbor resize; ignore 255',
              'prediction': 'per-pixel argmax; no threshold',
              'metrics': 'confusion matrix over valid pixels; omit class only if absent in truth and prediction',
              'selection_reason': 'Best eligible scratch-trained candidate on clean validation; pretrained weight redistribution terms unresolved.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f: json.dump(frozen, f, indent=2); f.write('\n')
    print(json.dumps(frozen, indent=2))

if __name__ == '__main__': main()
