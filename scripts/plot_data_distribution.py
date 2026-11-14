import json
from pathlib import Path
import matplotlib.pyplot as plt

summary = json.loads(Path('data/manifests/prepared_summary.json').read_text())
labels = ['Soil', 'Bedrock', 'Sand', 'Big rock']
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, split in zip(axes, ('train', 'val')):
    counts = summary['pixel_counts'][split]
    valid = sum(counts[str(i)] for i in range(4))
    percentages = [100 * counts[str(i)] / valid for i in range(4)]
    ax.bar(labels, percentages, color=['#b87039', '#7b8795', '#c7a94d', '#82505e'])
    ax.set_title(f'{split.capitalize()} • valid pixels')
    ax.set_ylabel('Percent')
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', labelrotation=20)
    for i, p in enumerate(percentages): ax.text(i, p + 1, f'{p:.1f}%', ha='center', fontsize=8)
fig.tight_layout()
out = Path('docs/assets/class_distribution.png'); out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
