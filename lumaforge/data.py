"""Training dataset backed by the consolidated, memory-mapped SIDD-CC crop cache."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .degrade import random_degradation
from .hybrid import prepare
from .isp.pipeline import ISPConfig

RAW_SCALE = 30000.0


class CropDataset(Dataset):
    """Yields (packed, noise, base, target, degraded-flag) per crop.

    Horizontal and vertical flips change the CFA phase, so each flip drops one sample to
    return to RGGB before the final crop.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        crop: int = 192,
        augment: bool = True,
        degrade: bool = True,
        isp: ISPConfig | None = None,
        epoch_size: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.index = json.loads((self.root / f"{split}.json").read_text())
        self.bayer = None
        self.rgb = None
        self.crop = crop
        self.augment = augment
        self.degrade = degrade
        self.isp = isp
        self.epoch_size = epoch_size or len(self.index)

    def _open(self) -> None:
        # Opened lazily so that spawned workers share pages via the OS file cache.
        if self.bayer is None:
            self.bayer = np.load(self.root / "bayer.npy", mmap_mode="r")
            self.rgb = np.load(self.root / "rgb.npy", mmap_mode="r")

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, item: int):
        self._open()
        rng = np.random.default_rng() if self.augment else np.random.default_rng(item)
        entry = self.index[item % len(self.index)]
        bayer = np.asarray(self.bayer[entry["row"]], dtype=np.float32) / RAW_SCALE
        rgb = np.asarray(self.rgb[entry["row"]], dtype=np.float32) / 255.0
        camera = entry["camera"]

        if self.augment:
            if rng.random() < 0.5:
                bayer, rgb = bayer[:, ::-1][:, 1:-1], rgb[:, ::-1][:, 1:-1]
            if rng.random() < 0.5:
                bayer, rgb = bayer[::-1][1:-1], rgb[::-1][1:-1]
            height, width = bayer.shape
            row = int(rng.integers(0, (height - self.crop) // 2 + 1)) * 2
            col = int(rng.integers(0, (width - self.crop) // 2 + 1)) * 2
        else:
            row = (bayer.shape[0] - self.crop) // 2
            col = (bayer.shape[1] - self.crop) // 2
            row, col = row - row % 2, col - col % 2
        bayer = bayer[row : row + self.crop, col : col + self.crop].copy()
        rgb = rgb[row : row + self.crop, col : col + self.crop].copy()

        compensation = 1.0
        if self.degrade:
            bayer, log = random_degradation(bayer, rng)
            compensation = 1.0 / log.get("exposure", 1.0)
        inputs = prepare(bayer, camera, self.isp, compensation)
        return (
            torch.from_numpy(inputs.packed),
            torch.from_numpy(inputs.noise),
            torch.from_numpy(inputs.base),
            torch.from_numpy(rgb.transpose(2, 0, 1)),
        )
