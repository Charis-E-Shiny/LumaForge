"""NumPy-only quality scores for the live API, which cannot bundle scikit-image.

The published evaluation uses CIEDE2000 from :mod:`lumaforge.metrics`. The live demo reports
the simpler CIE76 Delta E, and the UI labels it as such.
"""
from __future__ import annotations

import numpy as np

from .isp.classical import SRGB_TO_XYZ, srgb_degamma

_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((prediction - target) ** 2))
    return 99.0 if mse <= 1e-12 else 10.0 * float(np.log10(1.0 / mse))


def _lab(rgb: np.ndarray) -> np.ndarray:
    xyz = srgb_degamma(rgb) @ SRGB_TO_XYZ.T / _WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack((116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])), axis=-1)


def delta_e_76(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(_lab(prediction) - _lab(target), axis=-1)))
