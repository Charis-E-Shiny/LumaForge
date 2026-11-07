"""Invert the display pipeline so an ordinary RGB image can enter the ISP as a mosaic.

An uploaded JPEG or PNG has no RAW data. This produces an *approximate* sensor capture so the
demo can process it, and it is labelled as simulated wherever it is shown. The method follows
the idea of Brooks et al. (CVPR 2019): undo gamma, undo the colour matrix, undo white balance,
then keep only the sample each Bayer site would have recorded.
"""
from __future__ import annotations

import numpy as np

from .isp import classical as c


def unprocess(rgb: np.ndarray, ccm: np.ndarray, gains: tuple[float, float] = (2.0, 1.7)) -> np.ndarray:
    """Convert display-referred RGB (H, W, 3) in [0, 1] into an RGGB mosaic."""
    height, width = rgb.shape[:2]
    rgb = rgb[: height - height % 2, : width - width % 2]
    linear = c.srgb_degamma(rgb.astype(np.float32))
    camera = np.clip(linear @ np.linalg.inv(ccm).T, 0.0, 1.0).astype(np.float32)
    camera = camera * np.array([1.0 / gains[0], 1.0, 1.0 / gains[1]], dtype=np.float32)
    mosaic = np.empty(camera.shape[:2], dtype=np.float32)
    mosaic[0::2, 0::2] = camera[0::2, 0::2, 0]
    mosaic[0::2, 1::2] = camera[0::2, 1::2, 1]
    mosaic[1::2, 0::2] = camera[1::2, 0::2, 1]
    mosaic[1::2, 1::2] = camera[1::2, 1::2, 2]
    return mosaic
