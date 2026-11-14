import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import unittest
import torch
from marsterrain.losses import segmentation_loss
from marsterrain.models import UNet
from marsterrain.train import sample_records


class TrainingTest(unittest.TestCase):
    def test_forward_backward_and_ignore(self):
        torch.manual_seed(4); torch.set_num_threads(1)
        model = UNet(channels=4)
        images = torch.rand(2, 3, 32, 32)
        labels = torch.randint(0, 4, (2, 32, 32)); labels[:, :4] = 255
        logits = model(images)
        self.assertEqual(tuple(logits.shape), (2, 4, 32, 32))
        loss = segmentation_loss(logits, labels)
        altered = logits.detach().clone(); altered[:, :, :4] = 10000
        self.assertAlmostEqual(loss.item(), segmentation_loss(altered, labels).item(), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(model.output.weight.grad).all())
        with self.assertRaises(ValueError): segmentation_loss(logits, torch.full_like(labels, 255))

    def test_seeded_subset(self):
        rows = [{'id': str(i), 'split': 'train'} for i in range(20)]
        self.assertEqual(sample_records(rows, 'train', 4, 42), sample_records(rows, 'train', 4, 42))
        self.assertNotEqual(sample_records(rows, 'train', 4, 42), sample_records(rows, 'train', 4, 43))
