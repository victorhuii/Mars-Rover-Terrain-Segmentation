import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from PIL import Image
from marsterrain.predict import segment_image


class PredictionTest(unittest.TestCase):
    def test_real_model_rectangular_image_and_coverage(self):
        image = Image.new('RGB', (80, 40), (140, 100, 75))
        result = segment_image(image)
        self.assertEqual(result.mask.size, image.size)
        self.assertTrue(set(np.unique(np.asarray(result.mask))) <= {0, 1, 2, 3})
        self.assertAlmostEqual(sum(result.coverage.values()), 1.0)
        self.assertGreater(result.inference_ms, 0)

    def test_rejects_oversize_dimensions(self):
        with self.assertRaises(ValueError): segment_image(Image.new('L', (8193, 1)))

if __name__ == '__main__': unittest.main()
