"""Export trained checkpoints to ONNX, then apply reduced precision and quantization.

    abl_no_classical_denoise -> lumanet_compact.onnx (fp32), lumanet_compact_int8.onnx (deployed)
    learned                  -> lumanet_learned.onnx (fp32), lumanet_learned_int8.onnx (deployed)
    teacher                  -> lumanet_quality.onnx (fp32), lumanet_quality_fp16.onnx
    student                  -> lumanet_fast.onnx    (fp32), lumanet_fast_int8.onnx

Static int8 quantization is calibrated on validation crops. The output head and everything
after it stay in floating point, because the output is added to an 8-bit-accurate base image
and a coarsely quantized residual would show up directly as banding.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
import torch
from onnxconverter_common import float16
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import load_model

from lumaforge.data import CropDataset
from lumaforge.isp.pipeline import ISPConfig


def export(model: torch.nn.Module, path: Path) -> None:
    packed = torch.rand(1, 4, 128, 128)
    noise = torch.rand(1, 1, 128, 128)
    base = torch.rand(1, 3, 256, 256)
    axes = {"packed": {2: "h", 3: "w"}, "noise": {2: "h", 3: "w"},
            "base": {2: "h2", 3: "w2"}, "rgb": {2: "h2", 3: "w2"}}
    torch.onnx.export(model, (packed, noise, base), str(path), input_names=["packed", "noise", "base"],
                      output_names=["rgb"], dynamic_axes=axes, opset_version=17, dynamo=False)


class Reader(CalibrationDataReader):
    def __init__(self, dataset: CropDataset, names: set[str], samples: int = 48) -> None:
        self.dataset, self.names, self.count, self.index = dataset, names, samples, 0

    def get_next(self):
        if self.index >= self.count:
            return None
        packed, noise, base, _ = self.dataset[self.index * 7]
        self.index += 1
        values = {"packed": packed, "noise": noise, "base": base}
        return {k: v.numpy()[None] for k, v in values.items() if k in self.names}


def quantize(source: Path, target: Path, data: Path, isp: ISPConfig | None = None) -> None:
    prepared = source.with_suffix(".pre.onnx")
    quant_pre_process(str(source), str(prepared), skip_symbolic_shape=True)
    graph = onnx.load(str(prepared)).graph
    names = {i.name for i in graph.input}
    excluded = _nodes_after_head(graph)
    dataset = CropDataset(data, "val", 128, augment=False, degrade=True, isp=isp or ISPConfig())
    quantize_static(
        str(prepared), str(target), Reader(dataset, names), quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
        op_types_to_quantize=["Conv", "Relu", "Concat", "Add"], nodes_to_exclude=excluded,
    )
    prepared.unlink()


def _nodes_after_head(graph: onnx.GraphProto) -> list[str]:
    """Everything from the last Conv onwards, plus the colour head, stays floating point."""
    last_conv = max(i for i, node in enumerate(graph.node) if node.op_type == "Conv")
    excluded = [node.name for node in graph.node[last_conv:]]
    excluded += [node.name for node in graph.node if "color_head" in node.name]
    return excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--data", type=Path, default=Path("data/crops"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)

    quality, _ = load_model(args.checkpoints / "teacher.pt")
    export(quality, args.out / "lumanet_quality.onnx")
    half = float16.convert_float_to_float16(onnx.load(str(args.out / "lumanet_quality.onnx")), keep_io_types=True)
    onnx.save(half, str(args.out / "lumanet_quality_fp16.onnx"))

    fast, _ = load_model(args.checkpoints / "student.pt")
    export(fast, args.out / "lumanet_fast.onnx")
    quantize(args.out / "lumanet_fast.onnx", args.out / "lumanet_fast_int8.onnx", args.data)

    compact, _ = load_model(args.checkpoints / "abl_no_classical_denoise.pt")
    export(compact, args.out / "lumanet_compact.onnx")
    quantize(args.out / "lumanet_compact.onnx", args.out / "lumanet_compact_int8.onnx", args.data,
             ISPConfig(denoise=False))

    learned, _ = load_model(args.checkpoints / "learned.pt")
    export(learned, args.out / "lumanet_learned.onnx")
    quantize(args.out / "lumanet_learned.onnx", args.out / "lumanet_learned_int8.onnx", args.data)

    for path in sorted(args.out.glob("*.onnx")):
        print(f"{path.name}: {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
