"""ONNX inference runtime used by the deployed API. No PyTorch dependency."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .hybrid import prepare
from .isp.pipeline import ISPConfig

MODEL_DIR = Path(os.environ.get("LUMAFORGE_MODELS", Path(__file__).resolve().parents[1] / "models"))
MODEL_FILES = {
    "fast": "lumanet_compact_int8.onnx",
    "quality": "lumanet_learned_int8.onnx",
    "learned": "lumanet_learned_int8.onnx",
}
# The compact hybrid was trained on an undenoised base; it replaces the classical denoiser.
COMPACT_ISP = ISPConfig(denoise=False)
_sessions: dict[str, ort.InferenceSession] = {}


def session(name: str) -> ort.InferenceSession:
    """Lazily create one session per model; warm serverless instances reuse them."""
    file = MODEL_FILES[name]
    if file not in _sessions:
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.environ.get("LUMAFORGE_THREADS", "2"))
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _sessions[file] = ort.InferenceSession(
            str(MODEL_DIR / file), options, providers=["CPUExecutionProvider"]
        )
    return _sessions[file]


def infer(name: str, packed: np.ndarray, noise: np.ndarray, base: np.ndarray) -> np.ndarray:
    # The learned arm has no classical base image, so feed only what each graph declares.
    model = session(name)
    wanted = {i.name for i in model.get_inputs()}
    feed = {"packed": packed[None], "noise": noise[None], "base": base[None]}
    output = model.run(["rgb"], {k: v for k, v in feed.items() if k in wanted})[0]
    return np.clip(output[0].transpose(1, 2, 0), 0.0, 1.0).astype(np.float32)


@dataclass
class Rendered:
    traditional: np.ndarray
    learned: np.ndarray | None
    hybrid: np.ndarray
    route: str
    model: str
    stats: dict
    controls: dict
    gains: list[float]
    timings_ms: dict[str, float]


def render(
    bayer: np.ndarray,
    camera: str | None = None,
    mode: str = "auto",
    include_learned: bool = True,
    compensation: float = 1.0,
) -> Rendered:
    """Render one RGGB mosaic through all arms of the comparison.

    ``mode`` is "auto" (use the adaptive route), "fast" or "quality".
    """
    timings: dict[str, float] = {}
    started = time.perf_counter()
    inputs = prepare(bayer, camera, ISPConfig(), compensation)
    timings["classical"] = (time.perf_counter() - started) * 1000
    result = inputs.result

    route = result.controls.route if mode == "auto" else mode
    started = time.perf_counter()
    guided = prepare(bayer, camera, COMPACT_ISP, compensation) if route == "fast" else inputs
    hybrid = infer(route, guided.packed, guided.noise, guided.base)
    timings["hybrid_net"] = (time.perf_counter() - started) * 1000

    learned = None
    if include_learned:
        started = time.perf_counter()
        learned = infer("learned", inputs.packed, inputs.noise, inputs.base)
        timings["learned_net"] = (time.perf_counter() - started) * 1000

    return Rendered(
        traditional=result.rgb,
        learned=learned,
        hybrid=hybrid,
        route=route,
        model=MODEL_FILES[route],
        stats=result.stats.to_dict(),
        controls=result.controls.to_dict(),
        gains=[float(g) for g in result.gains],
        timings_ms={k: round(v, 1) for k, v in timings.items()},
    )
