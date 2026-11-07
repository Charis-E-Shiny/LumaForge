"""Image-quality metrics and differentiable training losses."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((prediction - target) ** 2))
    return 99.0 if mse <= 1e-12 else 10.0 * float(np.log10(1.0 / mse))


def ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(structural_similarity(prediction, target, channel_axis=-1, data_range=1.0))


def delta_e(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean CIEDE2000 between sRGB images (D65)."""
    return float(np.mean(deltaE_ciede2000(rgb2lab(target), rgb2lab(prediction))))


class LpipsMetric:
    """LPIPS (AlexNet); inputs are HWC images in [0, 1]."""

    def __init__(self) -> None:
        import lpips

        self.model = lpips.LPIPS(net="alex", verbose=False).eval()

    @torch.no_grad()
    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        a = torch.from_numpy(prediction.transpose(2, 0, 1))[None].float() * 2 - 1
        b = torch.from_numpy(target.transpose(2, 0, 1))[None].float() * 2 - 1
        return float(self.model(a, b).item())


# ---------------------------------------------------------------------------- losses
def _srgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    rgb = rgb.clamp(1e-4, 1.0)
    linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    matrix = rgb.new_tensor(
        [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
    )
    xyz = torch.einsum("ij,bjhw->bihw", matrix, linear)
    xyz = xyz / rgb.new_tensor([0.95047, 1.0, 1.08883]).view(1, 3, 1, 1)
    f = torch.where(xyz > 0.008856, xyz.clamp_min(1e-6) ** (1 / 3), 7.787 * xyz + 16 / 116)
    return torch.stack(
        (116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])), dim=1
    )


def ssim_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    kernel = 7
    c1, c2 = 0.01**2, 0.03**2
    mu_p = F.avg_pool2d(prediction, kernel, 1)
    mu_t = F.avg_pool2d(target, kernel, 1)
    var_p = F.avg_pool2d(prediction * prediction, kernel, 1) - mu_p**2
    var_t = F.avg_pool2d(target * target, kernel, 1) - mu_t**2
    cov = F.avg_pool2d(prediction * target, kernel, 1) - mu_p * mu_t
    value = ((2 * mu_p * mu_t + c1) * (2 * cov + c2)) / ((mu_p**2 + mu_t**2 + c1) * (var_p + var_t + c2))
    return 1.0 - value.mean()


def isp_loss(prediction: torch.Tensor, target: torch.Tensor, color_weight: float = 0.02,
             ssim_weight: float = 0.15) -> torch.Tensor:
    charbonnier = torch.sqrt((prediction - target) ** 2 + 1e-6).mean()
    loss = charbonnier + ssim_weight * ssim_loss(prediction.clamp(0, 1), target)
    if color_weight > 0:
        loss = loss + color_weight * F.l1_loss(_srgb_to_lab(prediction), _srgb_to_lab(target)) / 10.0
    return loss
