import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import unittest
from marsterrain.metrics import confusion_matrix, summarize

class MetricsTest(unittest.TestCase):
    def test_hand_computed_and_ignore(self):
        cm = confusion_matrix([0, 0, 1, 1, 255], [0, 1, 1, 1, 3])
        result = summarize(cm)
        self.assertEqual(result['confusion_matrix'], [[1, 1, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(result['per_class_iou'], [0.5, 2/3, None, None])
        self.assertEqual(result['pixel_accuracy'], 0.75)
        self.assertEqual(result['valid_pixels'], 4)
        self.assertAlmostEqual(result['per_class_dice'][0], 2/3)
    def test_empty_and_invalid(self):
        self.assertIsNone(summarize(confusion_matrix([255], [3]))['mean_iou'])
        with self.assertRaises(ValueError): confusion_matrix([0], [4])
        with self.assertRaises(ValueError): confusion_matrix([0, 1], [0])
