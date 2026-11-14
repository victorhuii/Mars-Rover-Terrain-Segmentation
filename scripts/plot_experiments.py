import json
from pathlib import Path
import matplotlib.pyplot as plt

experiments = [
    ('baseline-v1', 'U-Net, 512 images'),
    ('mobilenet-v1', 'Pretrained compact, 512 images'),
    ('mobile-scratch-v1', 'Scratch compact, 512 images'),
    ('mobile-scratch-v2', 'Scratch continuation, 2,048 images'),
    ('mobile-scratch-v3', 'Scratch continuation, 4,096 images'),
]
fig, ax = plt.subplots(figsize=(8, 4.5))
for directory, label in experiments:
    file = Path('experiments') / directory / 'history.json'
    if not file.exists(): continue
    rows = json.loads(file.read_text())
    ax.plot([row['epoch'] for row in rows], [row['validation']['mean_iou'] for row in rows], marker='o', markersize=3, label=label)
ax.set_xlabel('Epoch within experiment'); ax.set_ylabel('Validation subset mean IoU'); ax.set_ylim(0, 1)
ax.grid(alpha=.2); ax.legend(fontsize=8); fig.tight_layout()
output = Path('docs/assets/validation_curves.png'); output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=150)
