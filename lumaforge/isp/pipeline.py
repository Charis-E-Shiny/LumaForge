"""Traditional ISP assembled from the stages in :mod:`classical`.

Its unsharpened render is also the base layer of the hybrid network.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import classical as c
from .analysis import AdaptiveControls, SceneStats, analyze, plan

CALIBRATION_PATH = Path(__file__).with_name("calibration.json")

# Sensible defaults until scripts/calibrate.py has run; replaced by fitted values.
_FALLBACK = {
    "tone": {"exposure": 1.0, "shoulder": 2.0, "toe": 1.0},
    "ccm": {"generic": [[1.6, -0.45, -0.15], [-0.25, 1.5, -0.25], [0.0, -0.55, 1.55]]},
}


def load_calibration() -> dict:
    if CALIBRATION_PATH.exists():
        return json.loads(CALIBRATION_PATH.read_text())
    return _FALLBACK


CALIBRATION = load_calibration()


def ccm_for(camera: str | None) -> np.ndarray:
    table = CALIBRATION["ccm"]
    return np.asarray(table.get(camera or "generic", table["generic"]), dtype=np.float32)


@dataclass
class ISPConfig:
    """Stage switches. Disabling one produces an ablation variant."""

    denoise: bool = True
    white_balance: bool = True
    ccm: bool = True
    tone: bool = True
    chroma: bool = True
    adaptive: bool = True


@dataclass
class ISPResult:
    rgb: np.ndarray  # display-referred, sharpened output of the traditional ISP
    base: np.ndarray  # same output before sharpening; the hybrid base layer
    stats: SceneStats
    controls: AdaptiveControls
    gains: np.ndarray


def run_classical(
    bayer: np.ndarray,
    camera: str | None = None,
    config: ISPConfig | None = None,
    gains: np.ndarray | None = None,
    compensation: float = 1.0,
) -> ISPResult:
    """Render an RGGB mosaic to display-referred sRGB in the [0, 1] range.

    ``compensation`` is known digital gain from capture metadata (1.0 means none).
    """
    config = config or ISPConfig()
    packed = c.pack(bayer)
    stats = analyze(packed)
    controls = plan(stats, compensation)
    if not config.adaptive:
        controls = AdaptiveControls(controls.exposure, 0.012, 0.018, 0.25, "fast", "fixed settings")

    working = c.denoise_packed(packed, controls.denoise_sigma) if config.denoise else packed
    if config.white_balance:
        if gains is None:
            priors = CALIBRATION.get("wb_prior", {})
            prior = priors.get(camera or "generic", priors.get("generic"))
            gains = c.estimate_white_balance(working, prior=prior, prior_weight=CALIBRATION.get("wb_weight", 0.0))
    else:
        gains = np.ones(3, dtype=np.float32)
    working = c.apply_white_balance(working, gains)
    linear = c.demosaic_malvar(c.unpack(working))
    if config.ccm:
        linear = c.apply_ccm(linear, ccm_for(camera))
    tone = CALIBRATION["tone"]
    if config.tone:
        linear = c.tone_map(linear, tone["exposure"] * controls.exposure, tone["shoulder"], tone["toe"])
    else:
        linear = linear * controls.exposure
    base = np.clip(c.srgb_gamma(linear), 0.0, 1.0).astype(np.float32)
    if config.chroma:
        base = np.clip(c.chroma_denoise(base, controls.chroma_sigma), 0.0, 1.0)
    rgb = np.clip(c.sharpen_luma(base, controls.sharpen, threshold=controls.denoise_sigma), 0.0, 1.0)
    return ISPResult(rgb.astype(np.float32), base.astype(np.float32), stats, controls, gains)
