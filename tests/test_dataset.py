import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from PIL import Image
from marsterrain.dataset import decode_labels, prepare_pair


class DataTests(unittest.TestCase):
    def test_mask_values_and_exclusions_survive_resize(self):
        image = Image.fromarray(np.zeros((2, 3), dtype=np.uint8))
        label = Image.fromarray(np.array([[0, 1, 2], [3, 255, 0]], dtype=np.uint8))
        rover = Image.fromarray(np.array([[0, 0, 1], [0, 0, 0]], dtype=np.uint8))
        distance = Image.fromarray(np.array([[0, 0, 0], [1, 0, 0]], dtype=np.uint8))
        resized, mask = prepare_pair(image, label, rover, distance, 6)
        self.assertEqual(resized.mode, 'RGB')
        a = np.asarray(mask)
        self.assertEqual(set(np.unique(a)), {0, 1, 255})
        self.assertEqual(a[0, 5], 255)
        self.assertEqual(a[5, 0], 255)

    def test_invalid_labels_dimensions_and_empty(self):
        with self.assertRaises(ValueError): decode_labels(Image.fromarray(np.array([[4]], dtype=np.uint8)))
        with self.assertRaises(ValueError): decode_labels(Image.fromarray(np.array([[[0, 1, 0]]], dtype=np.uint8)))
        image = Image.new('L', (2, 2)); empty = Image.new('L', (2, 2), 255); binary = Image.new('L', (2, 2))
        with self.assertRaises(ValueError): prepare_pair(image, empty, binary, binary)
        with self.assertRaises(ValueError): prepare_pair(image, binary, Image.new('L', (1, 1)), binary)
        with self.assertRaises(ValueError): prepare_pair(image, binary, Image.new('L', (2, 2), 2), binary)

    def test_rgb_labels_are_lossless(self):
        labels = np.array([[0, 1], [2, 255]], dtype=np.uint8)
        self.assertTrue(np.array_equal(decode_labels(Image.fromarray(np.repeat(labels[:, :, None], 3, axis=2))), labels))

if __name__ == '__main__': unittest.main()
