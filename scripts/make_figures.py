"""Generate the report figures in public/figures from results and test blocks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import AdaptiveArm, TestSet, TorchArm, classical_arm

from lumaforge.isp.pipeline import ISPConfig
from lumaforge.metrics_lite import psnr

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3de"
SERIES = {"traditional": "#2a78d6", "learned": "#eb6834", "hybrid_adaptive": "#1baf7a"}
MARKERS = {"traditional": "o", "learned": "s", "hybrid_adaptive": "D"}
NAMES = {"traditional": "Traditional", "learned": "Learned", "hybrid_adaptive": "Hybrid"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def robustness(summary: dict, out: Path) -> None:
    rows = [r for r in summary["robust"] if r["arms"].get("learned")][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for axis, key, title in ((axes[0], "psnr", "PSNR (dB), higher is better"),
                             (axes[1], "delta_e", "ΔE2000, lower is better")):
        for position, row in enumerate(rows):
            values = [row["arms"][arm][key] for arm in SERIES]
            axis.plot([min(values), max(values)], [position, position], color=GRID, lw=2, zorder=1)
        for arm, colour in SERIES.items():
            axis.scatter([r["arms"][arm][key] for r in rows], range(len(rows)), s=70, color=colour,
                         marker=MARKERS[arm], edgecolor=SURFACE, linewidth=1.5, label=NAMES[arm], zorder=2)
        axis.set_title(title, loc="left", fontsize=12, color=MUTED)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color(GRID)
        axis.tick_params(left=False)
        axis.grid(axis="x", color=GRID, lw=0.7)
        axis.set_axisbelow(True)
    axes[0].set_yticks(range(len(rows)), [r["label"] for r in rows])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "robustness.png", dpi=150)
    plt.close(fig)


def gallery(picks: list[tuple[int, int, str]], arms: dict, data: TestSet, path: Path, crop: int = 240) -> None:
    columns = list(arms) + ["Ground truth"]
    fig, axes = plt.subplots(len(picks), len(columns), figsize=(2.5 * len(columns), 2.72 * len(picks)), squeeze=False)
    for row, (s, b, caption) in enumerate(picks):
        bayer, target, camera = data.block(s, b)
        bayer = bayer[: crop, : crop]
        target = target[: crop, : crop]
        for col, name in enumerate(columns):
            axis = axes[row, col]
            image = target if name == "Ground truth" else arms[name](bayer, camera, 1.0)
            axis.imshow(np.clip(image, 0, 1))
            axis.set_xticks([])
            axis.set_yticks([])
            axis.spines[:].set_visible(False)
            note = "" if name == "Ground truth" else f"{psnr(image, target):.1f} dB"
            axis.set_xlabel(note, fontsize=10)
            if row == 0:
                axis.set_title(name, fontsize=12)
            if col == 0:
                axis.set_ylabel(caption, fontsize=9.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--data", type=Path, default=Path("data/sidd_cc/test"))
    parser.add_argument("--out", type=Path, default=Path("public/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary = json.loads((args.results / "summary.json").read_text())
    robustness(summary, args.out)

    rows = json.loads((args.results / "main.json").read_text())["rows"]
    data = TestSet(args.data)
    hybrid = AdaptiveArm(TorchArm(Path("checkpoints/abl_no_classical_denoise.pt")), TorchArm(Path("checkpoints/learned.pt")))
    arms = {"Traditional": classical_arm(ISPConfig()), "Learned": TorchArm(Path("checkpoints/learned.pt")),
            "Hybrid": hybrid}

    def row_key(row):
        return row["scene_index"], row["block"]

    # Representative comparison: blocks at the median of the hybrid's PSNR, one per camera,
    # so the figure shows typical behaviour rather than the best cases.
    median = float(np.median([r["hybrid_adaptive"]["psnr"] for r in rows]))
    typical, seen = [], set()
    for row in sorted(rows, key=lambda r: abs(r["hybrid_adaptive"]["psnr"] - median)):
        if row["camera"] not in seen and row["hybrid_adaptive"]["lpips"] > 0.05:
            seen.add(row["camera"])
            typical.append((*row_key(row), f"{row['camera']}, scene {row['scene']}"))
    gallery(typical[:4], arms, data, args.out / "comparison.png")

    worst = sorted(rows, key=lambda r: r["hybrid_adaptive"]["psnr"])
    picks, scenes = [], set()
    for row in worst:
        if row["scene"] not in scenes:
            scenes.add(row["scene"])
            picks.append((*row_key(row), f"{row['camera']}, scene {row['scene']}"))
    gallery(picks[:4], arms, data, args.out / "failures_worst.png")

    regressions = sorted(rows, key=lambda r: r["hybrid_adaptive"]["psnr"] - r["traditional"]["psnr"])
    picks, scenes = [], set()
    for row in regressions:
        if row["scene"] not in scenes:
            scenes.add(row["scene"])
            picks.append((*row_key(row), f"{row['camera']}, scene {row['scene']}"))
    gallery(picks[:3], arms, data, args.out / "failures_regressions.png")
    print("FIGURES_DONE")


if __name__ == "__main__":
    main()
