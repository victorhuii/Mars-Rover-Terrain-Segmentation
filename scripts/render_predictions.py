"""Render local validation examples from a trusted U-Net checkpoint."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from marsterrain.models import UNet
from marsterrain.train import PreparedDataset, sample_records

PALETTE = np.array([[196, 127, 73], [156, 165, 180], [231, 197, 116], [130, 80, 94]], dtype=np.uint8)


def render(mask):
    valid = mask != 255
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[valid] = PALETTE[mask[valid]]
    return Image.fromarray(rgb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/processed/pairs256'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['metadata']['config']
    records = sample_records(json.loads((args.data / 'prepared_manifest.json').read_text())['records'], 'val', 4, config['seed'] + 1)
    dataset = PreparedDataset(args.data, records, size=config['size'])
    model = UNet(channels=config['channels']); model.load_state_dict(checkpoint['model']); model.eval()
    size = config['size']; canvas = Image.new('RGB', (size * 4, (size + 22) * len(records)))
    draw = ImageDraw.Draw(canvas)
    with torch.no_grad():
        for i, record in enumerate(records):
            image, label = dataset[i]; pred = model(image.unsqueeze(0)).argmax(1)[0].numpy()
            original = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            overlay = (original.astype(np.float32) * .5 + PALETTE[pred].astype(np.float32) * .5).astype(np.uint8)
            y = i * (size + 22); draw.text((2, y + 3), record['id'], fill='white')
            for j, panel in enumerate((Image.fromarray(original), render(label.numpy()), render(pred), Image.fromarray(overlay))):
                canvas.paste(panel, (j * size, y + 22))
    args.output.parent.mkdir(parents=True, exist_ok=True); canvas.save(args.output)
    print(f'Saved {len(records)} validation predictions to {args.output}')

if __name__ == '__main__': main()
