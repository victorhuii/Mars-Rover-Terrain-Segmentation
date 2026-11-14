"""Export a trusted trained checkpoint and validate CPU ONNX logits on validation images."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .dataset import CLASSES
from .evaluate import sha256
from .models import make_model
from .train import PreparedDataset, sample_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/processed/pairs256'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=3)
    args = parser.parse_args()
    if args.samples < 1: parser.error('samples must be positive')
    if args.output.exists() or args.metadata.exists(): raise FileExistsError('Export output already exists')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['metadata']['overfit_validation_is_training'] or checkpoint['epoch'] < 1:
        raise ValueError('Refusing to export an overfit diagnostic or untrained checkpoint')
    config = checkpoint['metadata']['config']; manifest = args.data / 'prepared_manifest.json'
    if sha256(manifest) != checkpoint['metadata']['prepared_manifest_sha256']: raise ValueError('Prepared manifest mismatch')
    records = sample_records(json.loads(manifest.read_text())['records'], 'val', args.samples, config['seed'] + 1)
    if len(records) != args.samples: raise ValueError('Not enough validation samples')
    data = PreparedDataset(args.data, records, size=config['size'])
    model = make_model(config.get('architecture', 'unet'), channels=config['channels'], pretrained=False).cpu().eval()
    model.load_state_dict(checkpoint['model'])
    inputs = [data[i][0].unsqueeze(0) for i in range(len(records))]
    args.output.parent.mkdir(parents=True, exist_ok=True); args.metadata.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, inputs[0], str(args.output), input_names=['image'], output_names=['logits'],
                      opset_version=17, dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(str(args.output))
    session = ort.InferenceSession(str(args.output), providers=['CPUExecutionProvider'])
    parity = []
    with torch.no_grad():
        for record, x in zip(records, inputs):
            reference = model(x).numpy(); actual = session.run(['logits'], {'image': x.numpy()})[0]
            error = float(np.max(np.abs(reference - actual)))
            agreement = float(np.mean(np.argmax(reference, axis=1) == np.argmax(actual, axis=1)))
            if not np.allclose(reference, actual, rtol=1e-4, atol=1e-4) or agreement < 0.999:
                raise ValueError(f'ONNX parity failed for {record["id"]}: error={error}, agreement={agreement}')
            parity.append({'id': record['id'], 'max_absolute_error': error, 'argmax_agreement': agreement})
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError): commit, dirty = None, None
    metadata = {'classes': list(CLASSES), 'palette_rgb': [[196,127,73], [156,165,180], [231,197,116], [130,80,94]], 'ignore_index': 255, 'input_shape': [1, 3, config['size'], config['size']],
                'input_range': [0, 1], 'normalization': 'ImageNet mean/std inside model' if config.get('architecture') == 'mobilenet_v3_small' else 'none',
                'opset': 17, 'architecture': config.get('architecture', 'unet'), 'dataset_revision': '93dbb79de6cd234799750b6129fbfb3b197582fa',
                'checkpoint_sha256': sha256(args.checkpoint), 'prepared_manifest_sha256': checkpoint['metadata']['prepared_manifest_sha256'],
                'onnx_sha256': sha256(args.output), 'onnx_bytes': args.output.stat().st_size,
                'cloudflare_pages_asset_limit_bytes': 25 * 1024 * 1024,
                'fits_cloudflare_pages_asset_limit': args.output.stat().st_size <= 25 * 1024 * 1024,
                'git_commit': commit, 'code_dirty': dirty, 'parity': parity, 'rtol': 1e-4, 'atol': 1e-4}
    args.metadata.write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))

if __name__ == '__main__': main()
