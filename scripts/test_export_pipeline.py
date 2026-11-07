"""Automated checks for the Member 2 model export contribution."""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


class ExportPipelineTests(unittest.TestCase):

    def read_script(self, filename):
        path = SCRIPTS / filename
        self.assertTrue(path.is_file(), f"Missing file: {filename}")
        return path.read_text(encoding="utf-8-sig")

    def test_export_script_syntax(self):
        source = self.read_script("export_onnx.py")
        ast.parse(source)

    def test_fp16_conversion_exists(self):
        source = self.read_script("export_onnx.py")
        self.assertIn("def convert_fp16(", source)
        self.assertIn("keep_io_types=True", source)

    def test_training_only_calibration(self):
        source = self.read_script("export_onnx.py")
        self.assertIn('"train"', source)
        self.assertIn("train.json", source)

    def test_int8_quantization_exists(self):
        source = self.read_script("export_onnx.py")
        self.assertIn("def quantize(", source)
        self.assertIn("quantize_static(", source)
        self.assertIn("QuantFormat.QDQ", source)

    def test_onnx_validation_script(self):
        source = self.read_script("validate_onnx.py")
        ast.parse(source)
        self.assertIn("np.testing.assert_allclose", source)
        self.assertIn("onnx.checker.check_model", source)

    def test_benchmark_script(self):
        source = self.read_script("benchmark_onnx.py")
        ast.parse(source)
        self.assertIn("perf_counter", source)
        self.assertIn("percentile", source)
        self.assertIn("Throughput", source)

    def test_calibration_reader_rewind(self):
        source = self.read_script("export_onnx.py")
        self.assertIn("def rewind(self):", source)

    def test_no_validation_calibration(self):
        source = self.read_script("export_onnx.py")
        self.assertNotIn('CropDataset(data, "val"', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)

