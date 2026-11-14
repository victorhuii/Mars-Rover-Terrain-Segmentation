import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PALETTE = np.array([[196, 127, 73], [156, 165, 180], [231, 197, 116], [130, 80, 94]], dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/processed/pairs256'))
    parser.add_argument('--output', type=Path, default=Path('data/processed/training_examples.png'))
    args = parser.parse_args()
    manifest = json.loads((args.data / 'prepared_manifest.json').read_text())
    entries = [entry for entry in manifest['records'] if entry['split'] == 'train'][:4]
    if not entries:
        raise ValueError('No prepared training images')
    width, height = manifest['size'], manifest['size']
    grid = Image.new('RGB', (width * 3, (height + 22) * len(entries)), (20, 20, 20))
    draw = ImageDraw.Draw(grid)
    for i, entry in enumerate(entries):
        image = Image.open(args.data / entry['image']).convert('RGB')
        mask = np.asarray(Image.open(args.data / entry['mask']))
        colors = np.zeros((height, width, 3), dtype=np.uint8)
        valid = mask != 255
        colors[valid] = PALETTE[mask[valid]]
        colored = Image.fromarray(colors)
        overlay = np.asarray(image).copy()
        overlay[valid] = (overlay[valid].astype(np.float32) * 0.5 + colors[valid].astype(np.float32) * 0.5).astype(np.uint8)
        y = i * (height + 22)
        draw.text((2, y + 3), entry['id'], fill='white')
        for j, panel in enumerate((image, colored, Image.fromarray(overlay))):
            grid.paste(panel, (j * width, y + 22))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)
    print(f'Saved {len(entries)} training examples to {args.output}')

if __name__ == '__main__': main()
