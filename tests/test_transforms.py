import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from PIL import Image
from marsterrain.transforms import transform_pair

class TransformTests(unittest.TestCase):
    def test_flip_is_synchronized_and_brightness_does_not_change_labels(self):
        source = np.array([[0, 1, 2, 255]], dtype=np.uint8)
        image, mask = transform_pair(Image.fromarray(source), Image.fromarray(source), flip=True, brightness=0.8)
        np.testing.assert_array_equal(np.asarray(mask), source[:, ::-1])
        self.assertEqual(np.asarray(image)[0, -1], 0)
    def test_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError): transform_pair(Image.new('L', (2, 2)), Image.new('L', (1, 1)))
        with self.assertRaises(ValueError): transform_pair(Image.new('L', (1, 1)), Image.new('L', (1, 1)), brightness=0)
