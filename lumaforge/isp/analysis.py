"""Blind scene analysis that drives the adaptive pipeline.

Nothing here needs capture metadata: noise and illumination are measured from the mosaic,
so the same code path works for a SIDD test block, an uploaded RAW, or a synthetic capture.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .classical import box_filter


@dataclass(frozen=True)
class SceneStats:
    shot: float  # variance per unit of signal (a in sigma^2 = a*I + b)
    read: float  # signal-independent variance (b)
    sigma: float  # noise std at 18% grey, the single number used for routing
    median: float  # median linear luminance
    p99: float  # 99th percentile linear luminance
    clipped: float  # fraction of saturated samples

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def estimate_noise_profile(packed: np.ndarray, bins: int = 12) -> tuple[float, float]:
    """Fit the heteroscedastic model sigma^2 = shot * I + read.

    The two green planes sample the same colour, so their difference cancels most image
    structure. A high-pass removes what is left, and the median absolute deviation inside
    each intensity bin rejects edges. Only the flatter pixels contribute.
    """
    green_a, green_b = packed[..., 1], packed[..., 2]
    difference = (green_a - green_b) / np.sqrt(2.0)
    residual = difference - box_filter(difference, 2)
    level = box_filter(0.5 * (green_a + green_b), 2)
    activity = box_filter(np.abs(residual), 2)
    flat = activity <= np.quantile(activity, 0.6)
    values, residuals = level[flat], residual[flat]
    edges = np.quantile(values, np.linspace(0.02, 0.98, bins + 1))
    xs, ys = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (values >= low) & (values < high)
        if inside.sum() < 32 or high <= low:
            continue
        sample = residuals[inside]
        mad = np.median(np.abs(sample - np.median(sample)))
        xs.append(float(np.mean(values[inside])))
        ys.append(float((1.4826 * mad) ** 2))
    if len(xs) < 3:
        variance = float((1.4826 * np.median(np.abs(residual))) ** 2)
        return 0.0, max(variance, 1e-9)
    x, y = np.asarray(xs), np.asarray(ys)
    # Weighted fit: the error of a variance estimate grows with the variance itself.
    weights = 1.0 / np.maximum(y, 1e-10)
    design = np.stack((x, np.ones_like(x)), axis=1) * weights[:, None]
    solution, *_ = np.linalg.lstsq(design, y * weights, rcond=None)
    shot, read = float(solution[0]), float(solution[1])
    if shot < 0.0:
        shot, read = 0.0, float(np.mean(y))
    return max(shot, 0.0), max(read, 1e-9)


def analyze(packed: np.ndarray) -> SceneStats:
    shot, read = estimate_noise_profile(packed)
    luma = 0.25 * packed.sum(axis=-1)
    subsample = luma[::2, ::2]
    return SceneStats(
        shot=shot,
        read=read,
        sigma=float(np.sqrt(shot * 0.18 + read)),
        median=float(np.median(subsample)),
        p99=float(np.quantile(subsample, 0.99)),
        clipped=float(np.mean(packed.max(axis=-1) >= 0.98)),
    )


def noise_map(packed: np.ndarray, stats: SceneStats) -> np.ndarray:
    """Per-pixel expected noise std, given to the network as a side input."""
    level = np.maximum(box_filter(packed.mean(axis=-1), 1), 0.0)
    return np.sqrt(stats.shot * level + stats.read).astype(np.float32)


@dataclass(frozen=True)
class AdaptiveControls:
    exposure: float
    denoise_sigma: float
    chroma_sigma: float
    sharpen: float
    route: str  # "fast" or "quality"
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def plan(
    stats: SceneStats,
    compensation: float = 1.0,
    noise_gate: float = 0.0284,  # set on validation by scripts/tune_gate.py
    max_sharpen: float = 0.25,
) -> AdaptiveControls:
    """Turn measurements into per-stage settings.

    ``compensation`` is the digital gain known from capture metadata. It is not estimated:
    SIDD references keep dark scenes dark, so a blind auto-exposure would brighten scenes
    that are meant to be dim. A phone knows its shutter and ISO, so taking this as an input
    is realistic; the blind case is reported as a failure mode.

    Gain multiplies noise, so routing follows the post-gain level. Sharpening backs off as
    noise rises because it would otherwise amplify grain.
    """
    exposure = float(np.clip(compensation, 1.0, 16.0))
    effective = stats.sigma * exposure
    sharpen = float(np.clip(max_sharpen * (1.0 - effective / 0.05), 0.0, max_sharpen))
    if effective >= noise_gate:
        route = "quality"
        reason = f"post-gain noise {effective:.4f} is above the gate: full-size model"
    else:
        route = "fast"
        reason = f"post-gain noise {effective:.4f} is below the gate: compact model"
    return AdaptiveControls(exposure, stats.sigma, stats.sigma * 1.5, sharpen, route, reason)
