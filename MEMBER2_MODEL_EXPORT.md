# Member 2 – Model Export and Optimization

## 1. Contribution Overview

Member 2 is responsible for model export, reduced-precision
conversion, quantization, validation, benchmarking and testing.

The objective is to prepare trained LumaForge models for
efficient ONNX Runtime inference.

## 2. ONNX Model Export

The export pipeline converts trained PyTorch checkpoints
into ONNX models.

Export configuration:
- ONNX opset: 17
- Dynamic spatial dimensions
- Input tensors: packed, noise and base
- Output tensor: rgb

## 3. FP16 Conversion

The FP16 conversion function uses onnxconverter-common.

Important features:
- Converts floating-point weights to FP16.
- Preserves input and output types.
- Validates the converted ONNX graph.
- Reports the generated model size.

## 4. INT8 Quantization

Static INT8 quantization uses ONNX Runtime.

Configuration:
- Quantization format: QDQ
- Activation type: QUInt8
- Weight type: QInt8
- Per-channel quantization: Enabled
- Calibration split: Training only
- Calibration samples: Up to 48

The validation and test splits are not used for calibration.

## 5. ONNX Validation

The validation script compares PyTorch and ONNX outputs.

It reports:
- Output dimensions
- Maximum absolute difference
- Mean absolute difference

Numerical consistency is checked using configurable
relative and absolute tolerances.

## 6. Performance Benchmarking

The benchmark script measures:
- ONNX model size
- Average inference latency
- Median inference latency
- P95 inference latency
- Throughput in FPS

The current benchmark configuration uses CPUExecutionProvider.

## 7. Automated Testing

The automated test suite checks:
- Export script syntax
- FP16 conversion implementation
- Training-only calibration
- INT8 quantization implementation
- ONNX validation script
- Benchmark implementation
- Calibration reader rewind
- Absence of validation-split calibration

## 8. Project Files

scripts/
- export_onnx.py
- validate_onnx.py
- benchmark_onnx.py
- test_export_pipeline.py

## 9. Current Limitations

Actual model conversion and benchmarking require:
- Trained PyTorch checkpoints
- Dataset split JSON files
- Bayer and RGB NumPy cache files
- Required Python dependencies
- Exported ONNX models

Source-level implementation and automated checks
do not replace real model execution.

## 10. Future Work

- Export trained teacher, student and learned models.
- Generate and validate FP16 models.
- Generate INT8 models using training-only calibration.
- Compare PyTorch and ONNX numerical outputs.
- Benchmark model size and inference performance.
- Record actual experimental results.

## 11. Deliverables

Member 2 deliverables include the export pipeline,
FP16 conversion, INT8 quantization, ONNX validation,
performance benchmarking, automated tests and
technical documentation.
