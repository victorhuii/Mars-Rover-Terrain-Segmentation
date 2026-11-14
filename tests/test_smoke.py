import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


class PackageSmokeTest(unittest.TestCase):
    def test_package_imports(self):
        import marsterrain

        self.assertEqual(pathlib.Path(marsterrain.__file__).name, "__init__.py")
