"""Classical ISP stages implemented in pure NumPy.

Everything here runs inside the Vercel function as well as in the training data loader, so
it deliberately has no OpenCV, SciPy or Torch dependency. One implementation serves every
environment, which rules out train/serve skew in the classical branch.

Array conventions:
    * Bayer mosaics are ``float32`` (H, W), black-subtracted, 1.0 = sensor white, RGGB phase.
    * Packed RAW is ``float32`` (H/2, W/2, 4) in R, Gr, Gb, B order.
    * RGB images are ``float32`` (H, W, 3).
"""
from __future__ import annotations

import numpy as np

# Linear sRGB -> CIE XYZ (D65), used to derive camera-to-sRGB matrices.
SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)

# BT.601 full-range transform. Luma and chroma are filtered separately because the eye is
# far less sensitive to chroma resolution, so chroma can be denoised harder.
RGB_TO_YCBCR = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ],
    dtype=np.float32,
)
YCBCR_TO_RGB = np.linalg.inv(RGB_TO_YCBCR).astype(np.float32)


# --------------------------------------------------------------------------- Bayer helpers
def to_rggb(bayer: np.ndarray, cfa: str) -> np.ndarray:
    """Crop a mosaic of any 2x2 CFA so that its top-left sample is red.

    Blue-first patterns shift by one sample, so the output can be up to two pixels smaller.
    """
    cfa = cfa.lower()
    if sorted(cfa) != ["b", "g", "g", "r"]:
        raise ValueError(f"unsupported CFA pattern: {cfa!r}")
    index = cfa.index("r")
    out = bayer[index // 2 :, index % 2 :]
    height, width = out.shape
    return out[: height - height % 2, : width - width % 2]


def pack(bayer: np.ndarray) -> np.ndarray:
    """RGGB mosaic -> (H/2, W/2, 4) stack."""
    return np.stack((bayer[0::2, 0::2], bayer[0::2, 1::2], bayer[1::2, 0::2], bayer[1::2, 1::2]), axis=-1)


def unpack(packed: np.ndarray) -> np.ndarray:
    """(H, W, 4) stack -> RGGB mosaic of size (2H, 2W)."""
    height, width = packed.shape[:2]
    bayer = np.empty((height * 2, width * 2), dtype=packed.dtype)
    bayer[0::2, 0::2] = packed[..., 0]
    bayer[0::2, 1::2] = packed[..., 1]
    bayer[1::2, 0::2] = packed[..., 2]
    bayer[1::2, 1::2] = packed[..., 3]
    return bayer


# ------------------------------------------------------------------------------- Filtering
def box_filter(image: np.ndarray, radius: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window using summed-area tables; cost independent of radius."""
    if radius <= 0:
        return image.copy()
    size = 2 * radius + 1
    pad = [(radius + 1, radius)] * 2 + [(0, 0)] * (image.ndim - 2)
    padded = np.pad(image.astype(np.float64), pad, mode="reflect")
    table = padded.cumsum(axis=0).cumsum(axis=1)
    out = table[size:, size:] - table[:-size, size:] - table[size:, :-size] + table[:-size, :-size]
    return (out / (size * size)).astype(np.float32)


def guided_filter(guide: np.ndarray, source: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Edge-preserving smoothing (He et al., TPAMI 2013) with a single-channel guide."""
    guide_mean = box_filter(guide, radius)
    source_mean = box_filter(source, radius)
    if source.ndim == 3:
        guide_mean_x = guide_mean[..., None]
        guide_x = guide[..., None]
    else:
        guide_mean_x = guide_mean
        guide_x = guide
    covariance = box_filter(guide_x * source, radius) - guide_mean_x * source_mean
    variance = box_filter(guide * guide, radius) - guide_mean * guide_mean
    variance = variance[..., None] if source.ndim == 3 else variance
    a = covariance / (variance + eps)
    b = source_mean - a * guide_mean_x
    return box_filter(a, radius) * guide_x + box_filter(b, radius)


def denoise_packed(packed: np.ndarray, sigma: float) -> np.ndarray:
    """Raw-domain denoise, guided by the mean green plane.

    ``eps`` scales with the noise variance, so a clean capture passes almost untouched and
    a high-ISO capture gets strong smoothing. Luminance-dependent shot noise is handled by
    variance stabilising the signal with a square root before filtering.
    """
    if sigma <= 1e-5:
        return packed
    offset = 2.0 * sigma
    stabilised = np.sqrt(np.maximum(packed + offset, 0.0))
    guide = stabilised[..., 1:3].mean(axis=-1)
    # Noise in sqrt space is roughly sigma / (2 * sqrt(signal)); 0.18 is mid-grey.
    stabilised_sigma = sigma / (2.0 * np.sqrt(0.18 + offset))
    eps = float(np.clip(4.0 * stabilised_sigma**2, 1e-6, 1e-2))
    radius = 1 if sigma < 0.01 else 2
    filtered = guided_filter(guide, stabilised, radius, eps)
    return np.square(filtered) - offset


def demosaic_malvar(bayer: np.ndarray) -> np.ndarray:
    """Gradient-corrected bilinear demosaicing (Malvar, He & Cutler, ICASSP 2004)."""
    padded = np.pad(bayer, 2, mode="reflect")

    def tap(dy: int, dx: int) -> np.ndarray:
        height, width = padded.shape
        return padded[2 + dy : height - 2 + dy, 2 + dx : width - 2 + dx]

    centre = tap(0, 0)
    n1, s1, w1, e1 = tap(-1, 0), tap(1, 0), tap(0, -1), tap(0, 1)
    n2, s2, w2, e2 = tap(-2, 0), tap(2, 0), tap(0, -2), tap(0, 2)
    nw, ne, sw, se = tap(-1, -1), tap(-1, 1), tap(1, -1), tap(1, 1)

    green_at_rb = (4 * centre + 2 * (n1 + s1 + w1 + e1) - (n2 + s2 + w2 + e2)) / 8
    same_row = (5 * centre + 4 * (w1 + e1) - (w2 + e2 + nw + ne + sw + se) + 0.5 * (n2 + s2)) / 8
    same_col = (5 * centre + 4 * (n1 + s1) - (n2 + s2 + nw + ne + sw + se) + 0.5 * (w2 + e2)) / 8
    opposite = (6 * centre + 2 * (nw + ne + sw + se) - 1.5 * (n2 + s2 + w2 + e2)) / 8

    height, width = bayer.shape
    r_row = np.zeros((height, width), dtype=bool)
    r_row[0::2] = True
    r_col = np.zeros((height, width), dtype=bool)
    r_col[:, 0::2] = True
    red = r_row & r_col
    blue = ~r_row & ~r_col
    green_r = r_row & ~r_col  # green on a red row
    green_b = ~r_row & r_col  # green on a blue row

    out = np.empty((height, width, 3), dtype=np.float32)
    out[..., 0] = np.select([red, green_r, green_b, blue], [centre, same_row, same_col, opposite])
    out[..., 1] = np.where(red | blue, green_at_rb, centre)
    out[..., 2] = np.select([blue, green_b, green_r, red], [centre, same_row, same_col, opposite])
    return out


# ------------------------------------------------------------------- Colour and white point
def estimate_white_balance(
    packed: np.ndarray,
    power: float = 6.0,
    prior: np.ndarray | None = None,
    prior_weight: float = 0.0,
) -> np.ndarray:
    """Shades-of-grey illuminant estimate; returns (R, G, B) gains with G = 1.

    Clipped and near-black pixels are excluded: clipped ones carry no colour, and near-black
    ones are dominated by noise. Gains are limited to the range a real illuminant produces.
    A small crop can be dominated by one object colour, so the estimate can be shrunk in log
    space towards a prior fitted on training scenes.
    """
    rgb = np.stack((packed[..., 0], packed[..., 1:3].mean(axis=-1), packed[..., 3]), axis=-1)
    rgb = box_filter(rgb, 2)  # suppress noise before the power mean
    peak = np.quantile(rgb[::2, ::2], 0.995)
    valid = (rgb.max(axis=-1) < 0.98) & (rgb.min(axis=-1) > max(0.01 * peak, 1e-4))
    samples = rgb[valid] if valid.sum() > 64 else np.maximum(rgb.reshape(-1, 3), 1e-5)
    illuminant = np.power(np.mean(np.power(np.maximum(samples, 1e-6), power), axis=0), 1.0 / power)
    gains = illuminant[1] / np.maximum(illuminant, 1e-6)
    if prior is not None and prior_weight > 0.0:
        log_gains = (1.0 - prior_weight) * np.log(np.maximum(gains, 1e-3))
        gains = np.exp(log_gains + prior_weight * np.log(np.asarray(prior, dtype=np.float64)))
    return np.clip(gains, 0.4, 4.0).astype(np.float32)


def apply_white_balance(packed: np.ndarray, gains: np.ndarray) -> np.ndarray:
    return packed * np.array([gains[0], gains[1], gains[1], gains[2]], dtype=np.float32)


def camera_to_srgb_matrix(color_matrix: np.ndarray) -> np.ndarray:
    """Build a CCM from a DNG ColorMatrix (XYZ -> camera).

    Each row is normalised to sum to one so that neutral stays neutral after white balance.
    Kept for cameras with no fitted calibration; SIDD cameras use fitted matrices.
    """
    xyz_to_camera = np.asarray(color_matrix, dtype=np.float64).reshape(3, 3)
    srgb_to_camera = xyz_to_camera @ SRGB_TO_XYZ
    srgb_to_camera = srgb_to_camera / srgb_to_camera.sum(axis=1, keepdims=True)
    return np.linalg.inv(srgb_to_camera).astype(np.float32)


def apply_ccm(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return rgb @ matrix.T


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    return rgb @ RGB_TO_YCBCR.T


def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    return ycbcr @ YCBCR_TO_RGB.T


def srgb_gamma(linear: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 transfer function."""
    linear = np.clip(linear, 0.0, None)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.power(linear, 1 / 2.4) - 0.055)


def srgb_degamma(encoded: np.ndarray) -> np.ndarray:
    encoded = np.clip(encoded, 0.0, 1.0)
    return np.where(encoded <= 0.04045, encoded / 12.92, np.power((encoded + 0.055) / 1.055, 2.4))


def tone_map(linear: np.ndarray, exposure: float, shoulder: float, toe: float) -> np.ndarray:
    """Global filmic curve applied to luminance so hue and saturation are preserved.

    ``shoulder`` rolls highlights off (extended Reinhard) and ``toe`` is a power applied to
    the normalised luminance, which lifts or deepens shadows.
    """
    scaled = np.maximum(linear * exposure, 0.0)
    luma = np.maximum(scaled @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), 1e-6)
    compressed = luma * (1.0 + luma / (shoulder * shoulder)) / (1.0 + luma)
    compressed = np.power(np.clip(compressed, 1e-6, None), toe)
    ratio = compressed / luma
    return scaled * ratio[..., None]


def sharpen_luma(rgb: np.ndarray, amount: float, radius: int = 1, threshold: float = 0.0) -> np.ndarray:
    """Unsharp mask on the Y channel only; avoids colour fringing around edges.

    Detail smaller than ``threshold`` is treated as noise and left alone.
    """
    if amount <= 0.0:
        return rgb
    ycbcr = rgb_to_ycbcr(rgb)
    luma = ycbcr[..., 0]
    detail = luma - guided_filter(luma, luma, radius, 1e-3)
    if threshold > 0.0:
        detail = detail * np.clip((np.abs(detail) - threshold) / max(threshold, 1e-6), 0.0, 1.0)
    ycbcr[..., 0] = luma + amount * detail
    return ycbcr_to_rgb(ycbcr)


def chroma_denoise(rgb: np.ndarray, sigma: float) -> np.ndarray:
    """Edge-preserving chroma smoothing in YCbCr, using luma as the guide."""
    if sigma <= 1e-5:
        return rgb
    ycbcr = rgb_to_ycbcr(rgb)
    radius = 2 if sigma < 0.01 else 4
    eps = float(np.clip(25.0 * sigma**2, 1e-5, 2e-2))
    ycbcr[..., 1:] = guided_filter(ycbcr[..., 0], ycbcr[..., 1:], radius, eps)
    return ycbcr_to_rgb(ycbcr)
