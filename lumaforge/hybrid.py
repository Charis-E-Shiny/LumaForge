"""Hybrid input preparation. NumPy only: shared by training, evaluation and the API."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .isp import classical as c
from .isp.analysis import noise_map
from .isp.pipeline import ISPConfig, ISPResult, run_classical

NOISE_SCALE = 20.0  # brings typical noise std into roughly [0, 1] for the network


@dataclass
class NetInputs:
    packed: np.ndarray  # (4, H/2, W/2)
    noise: np.ndarray  # (1, H/2, W/2)
    base: np.ndarray  # (3, H, W)
    result: ISPResult


def encode_packed(packed: np.ndarray, gains: np.ndarray, exposure: float) -> np.ndarray:
    """Apply gain, then a square root that compresses shot noise and dynamic range."""
    balanced = c.apply_white_balance(packed, gains) * exposure
    return np.sqrt(np.clip(balanced, 0.0, 4.0)).astype(np.float32)


def prepare(
    bayer: np.ndarray,
    camera: str | None = None,
    config: ISPConfig | None = None,
    compensation: float = 1.0,
) -> NetInputs:
    """Run the classical branch and build all network inputs for an RGGB mosaic."""
    # U-Net downsamples three times on the packed grid: full-size sides must divide by 16.
    height, width = bayer.shape
    bayer = bayer[: height - height % 16, : width - width % 16]
    result = run_classical(bayer, camera, config, compensation=compensation)
    packed = c.pack(bayer)
    encoded = encode_packed(packed, result.gains, result.controls.exposure)
    noise = noise_map(packed, result.stats) * NOISE_SCALE * result.controls.exposure
    return NetInputs(
        packed=encoded.transpose(2, 0, 1),
        noise=noise[None].astype(np.float32),
        base=result.base.transpose(2, 0, 1),
        result=result,
    )
