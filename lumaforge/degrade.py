"""Physically motivated degradations applied in the RAW domain.

They are used to augment training and to build the challenge conditions at evaluation time.
All operate on an RGGB mosaic, because that is where real sensors degrade: exposure and
colour cast scale the signal before noise is added, and noise is added before demosaicing.
"""
from __future__ import annotations

import numpy as np


def checker_gains(shape: tuple[int, int], red: float, blue: float) -> np.ndarray:
    gains = np.ones(shape, dtype=np.float32)
    gains[0::2, 0::2] = red
    gains[1::2, 1::2] = blue
    return gains


def underexpose(bayer: np.ndarray, factor: float) -> np.ndarray:
    return bayer * float(factor)


def add_sensor_noise(bayer: np.ndarray, shot: float, read: float, rng: np.random.Generator) -> np.ndarray:
    variance = shot * np.maximum(bayer, 0.0) + read
    return bayer + rng.normal(0.0, 1.0, bayer.shape).astype(np.float32) * np.sqrt(variance).astype(np.float32)


def colour_cast(bayer: np.ndarray, red: float, blue: float) -> np.ndarray:
    return bayer * checker_gains(bayer.shape, red, blue)


def low_light(bayer: np.ndarray, factor: float, rng: np.random.Generator) -> np.ndarray:
    """Fewer photons at the same sensor: signal drops, and read noise matters more.

    Scaling an existing capture also scales its noise, which would understate how noisy a
    genuinely dark frame is, so noise is added back in proportion to the exposure loss.
    """
    dark = underexpose(bayer, factor)
    return add_sensor_noise(dark, 3e-4 * (1.0 - factor), 1e-6 * (1.0 - factor**2), rng)


def random_degradation(bayer: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float]]:
    """Sample a training degradation. About a quarter of samples stay untouched."""
    log: dict[str, float] = {}
    out = bayer
    if rng.random() < 0.45:
        factor = float(2.0 ** -rng.uniform(0.5, 3.0))
        out = low_light(out, factor, rng)
        log["exposure"] = factor
    if rng.random() < 0.35:
        shot = float(10 ** rng.uniform(-4.0, -2.5))
        read = float(10 ** rng.uniform(-7.0, -4.5))
        out = add_sensor_noise(out, shot, read, rng)
        log["shot"], log["read"] = shot, read
    if rng.random() < 0.35:
        red = float(np.exp(rng.uniform(-0.4, 0.4)))
        blue = float(np.exp(rng.uniform(-0.4, 0.4)))
        out = colour_cast(out, red, blue)
        log["red"], log["blue"] = red, blue
    return out, log


# Named conditions used in the robustness evaluation. Fixed seeds keep them repeatable.
CONDITIONS: dict[str, dict] = {
    "clean": {},
    "low_light_1stop": {"exposure": 0.5},
    "low_light_2stop": {"exposure": 0.25},
    "low_light_3stop": {"exposure": 0.125},
    "noise_moderate": {"shot": 1e-3, "read": 3e-6},
    "noise_severe": {"shot": 5e-3, "read": 3e-5},
    "cast_warm": {"red": 1.45, "blue": 0.7},
    "cast_cool": {"red": 0.7, "blue": 1.45},
    "combined_severe": {"exposure": 0.25, "shot": 1e-3, "read": 3e-6, "red": 1.3, "blue": 0.75},
}


def apply_condition(bayer: np.ndarray, condition: dict, rng: np.random.Generator) -> np.ndarray:
    out = bayer
    if "exposure" in condition:
        out = low_light(out, condition["exposure"], rng)
    if "shot" in condition:
        out = add_sensor_noise(out, condition["shot"], condition["read"], rng)
    if "red" in condition:
        out = colour_cast(out, condition["red"], condition["blue"])
    return out
