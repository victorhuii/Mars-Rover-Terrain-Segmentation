"""Minimal repository setup smoke test; ML tests follow with the pipeline."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


class PackageSmokeTest(unittest.TestCase):
    def test_package_imports(self):
        import marsterrain

        self.assertIsNotNone(marsterrain.__doc__)
