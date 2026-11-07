"""Generate results/summary.json and eval_P2_T32.md from the raw evaluation outputs.

Every number in the report comes from results/*.json, so it cannot drift from the evidence.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

USNS = ["1RUA24CSE0100", "1RUA24CSE0109", "1RUA24CSE0111", "1RUA24CSE0114", "1RUA24CSE0098"]

ARMS = {
    "traditional_fixed": "Traditional ISP, fixed settings",
    "traditional": "Traditional ISP, adaptive (baseline)",
    "learned_fp32": "Learned ISP, fp32",
    "learned_int8": "Learned ISP, int8",
    "hybrid_compact_fp32": "Hybrid compact, fp32",
    "hybrid_compact_int8": "Hybrid compact, int8",
    "hybrid_quality_fp32": "Hybrid on denoised base, width 24, fp32",
    "hybrid_quality_fp16": "Hybrid on denoised base, width 24, fp16",
    "hybrid_fast_fp32": "Hybrid on denoised base, distilled, fp32",
    "hybrid_fast_int8": "Hybrid on denoised base, distilled, int8",
    "hybrid_adaptive": "Hybrid adaptive (deployed)",
}
CONDITIONS = {
    "clean": "Clean capture",
    "low_light_1stop": "Low light, 1 stop",
    "low_light_2stop": "Low light, 2 stops",
    "low_light_3stop": "Low light, 3 stops",
    "noise_moderate": "Added noise, moderate",
    "noise_severe": "Added noise, severe",
    "cast_warm": "Colour cast, warm",
    "cast_cool": "Colour cast, cool",
    "combined_severe": "Combined: 2 stops, noise, cast",
    "blind_underexposed_2stop": "2 stops, gain metadata missing",
}
ABLATION_CLASSICAL = {
    "full": "Full classical pipeline",
    "no_denoise": "No raw denoising",
    "no_white_balance": "No white balance",
    "no_ccm": "No colour correction matrix",
    "no_tone": "No tone mapping",
    "no_chroma_filter": "No chroma filter",
    "no_adaptive": "No adaptive control",
}
ABLATION_TRAINED = {
    "abl_base": "Hybrid, all components",
    "abl_no_noise_map": "No noise map input",
    "abl_no_color_head": "No global colour head",
    "abl_no_degrade": "No degradation training",
    "abl_no_color_loss": "No Lab colour loss",
    "abl_no_classical_denoise": "No classical denoise in base",
    "abl_learned": "No classical base (learned only)",
}


def metric_row(values: dict) -> dict:
    return {key: round(values[key], 4) for key in ("psnr", "ssim", "delta_e", "lpips")}


def metric_table(rows: list[tuple[str, dict]], highlight: str | None = None) -> str:
    lines = ["| Configuration | PSNR (dB) | SSIM | ΔE2000 | LPIPS |", "|---|---:|---:|---:|---:|"]
    for label, values in rows:
        cells = f"{values['psnr']:.2f} | {values['ssim']:.4f} | {values['delta_e']:.2f} | {values['lpips']:.4f}"
        name = f"**{label}**" if label == highlight else label
        lines.append(f"| {name} | {cells} |")
    return "\n".join(lines)


def paired_interval(rows: list[dict], a: str, b: str, key: str) -> tuple[float, float, float]:
    """Mean difference a-b with a 95% bootstrap CI, resampling test scenes, not blocks.

    Blocks within a scene are strongly correlated; scenes are the independent unit.
    """
    scenes: dict[str, list[float]] = {}
    for row in rows:
        scenes.setdefault(row["scene"], []).append(row[a][key] - row[b][key])
    means = np.array([np.mean(v) for v in scenes.values()])
    rng = np.random.default_rng(0)
    draws = rng.choice(means, size=(10000, len(means))).mean(axis=1)
    return float(means.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def group_breakdown(rows: list[dict], arms: list[str]) -> dict:
    groups: dict[str, dict[str, list]] = {}
    for row in rows:
        table = groups.setdefault(row["camera"], {a: [] for a in arms})
        for arm in arms:
            table[arm].append((row[arm]["psnr"], row[arm]["delta_e"]))
    return {cam: {a: [float(x) for x in np.mean(v, axis=0)] for a, v in table.items()} for cam, table in groups.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--eval", type=Path, default=Path("eval_P2_T32.md"))
    args = parser.parse_args()

    main_data = json.loads((args.results / "main.json").read_text())
    robust = json.loads((args.results / "robust.json").read_text())
    ablation = json.loads((args.results / "ablation.json").read_text())
    bench = json.loads((args.results / "benchmark.json").read_text())

    summary_main = main_data["summary"]
    rows = main_data["rows"]
    deltas = {
        key: paired_interval(rows, "hybrid_adaptive", "traditional", key)
        for key in ("psnr", "ssim", "delta_e", "lpips")
    }
    deltas_learned = {
        key: paired_interval(rows, "hybrid_adaptive", "learned_int8", key)
        for key in ("psnr", "ssim", "delta_e", "lpips")
    }
    by_camera = group_breakdown(rows, ["traditional", "learned_fp32", "hybrid_adaptive"])

    summary = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "test": {"blocks": main_data["blocks"], "scenes": len({r["scene"] for r in rows})},
        "routed": main_data["routed"],
        "main": [{"id": k, "label": v, **metric_row(summary_main[k])} for k, v in ARMS.items()],
        "deltas": {"traditional": deltas, "learned": deltas_learned},
        "robust": [
            {"id": k, "label": CONDITIONS[k], "blocks": robust[k]["blocks"],
             "arms": {a: metric_row(m) for a, m in robust[k]["summary"].items()}}
            for k in CONDITIONS if k in robust
        ],
        "ablation": {
            "classical": [{"id": k, "label": v, **metric_row(ablation["classical"]["summary"][k])}
                          for k, v in ABLATION_CLASSICAL.items()],
            "trained_clean": [{"id": k, "label": v, **metric_row(ablation["trained_clean"]["summary"][k])}
                              for k, v in ABLATION_TRAINED.items() if k in ablation["trained_clean"]["summary"]],
            "trained_severe": [{"id": k, "label": v, **metric_row(ablation["trained_severe"]["summary"][k])}
                               for k, v in ABLATION_TRAINED.items() if k in ablation["trained_severe"]["summary"]],
        },
        "benchmark": bench,
        "by_camera": by_camera,
    }
    (args.results / "summary.json").write_text(json.dumps(summary, indent=1))

    total = sum(main_data["routed"].values()) or 1
    quality_share = main_data["routed"]["quality"] / total
    machine = bench["machine"]
    model_lines = [
        "| Model | Parameters | GMACs at 512 px | File size (MB) |", "|---|---:|---:|---:|",
    ]
    latency_lines = [
        "| Model | 256 px, 1 thread (ms) | 512 px, 1 thread (ms) | 512 px, 4 threads (ms) | 1024 px, 4 threads (ms) | Peak memory at 512 px (MB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in bench["models"].items():
        label = ARMS[name]
        model_lines.append(f"| {label} | {row['params']:,} | {row['gmacs_512']:.2f} | {row['size_mb']:.2f} |")
        lat = row["latency"]
        latency_lines.append(
            f"| {label} | {lat['256px_1t']['ms_median']:.0f} | {lat['512px_1t']['ms_median']:.0f} | "
            f"{lat['512px_4t']['ms_median']:.0f} | {lat['1024px_4t']['ms_median']:.0f} | "
            f"{lat['512px_1t']['peak_mb']:.0f} |"
        )
    classical = bench["classical_ms"]

    def fmt(interval: tuple[float, float, float], digits: int) -> str:
        return f"{interval[0]:+.{digits}f} (95% CI {interval[1]:+.{digits}f} to {interval[2]:+.{digits}f})"

    robust_lines = [
        "| Condition | Blocks | Traditional PSNR | Learned PSNR | Hybrid PSNR | Traditional ΔE | Learned ΔE | Hybrid ΔE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["robust"]:
        arms = row["arms"]
        learned = arms.get("learned")
        robust_lines.append(
            f"| {row['label']} | {row['blocks']} | {arms['traditional']['psnr']:.2f} | "
            f"{'n/a' if learned is None else format(learned['psnr'], '.2f')} | "
            f"{arms['hybrid_adaptive']['psnr']:.2f} | {arms['traditional']['delta_e']:.2f} | "
            f"{'n/a' if learned is None else format(learned['delta_e'], '.2f')} | "
            f"{arms['hybrid_adaptive']['delta_e']:.2f} |"
        )

    text = f"""{",".join(USNS)}

# Evaluation: P2, Team T32 (Asphalt)

**Project:** LumaForge, a hybrid learned smartphone ISP.
**Test data:** {summary["test"]["blocks"]} blocks of 256 x 256 from {summary["test"]["scenes"]} held-out SIDD-CC scenes, unseen in training, calibration and tuning.
**Task:** noisy Bayer RAW to sRGB, scored against the dataset's 150-frame-averaged sRGB ground truth.
**White balance:** estimated blindly, because the test split ships without capture metadata.
**Deployed system:** "Hybrid adaptive". It sent {quality_share:.0%} of clean blocks to the learned int8 model (quality route) and the rest to the compact hybrid int8 model (fast route).
**Model selection:** the deployed pair was chosen after the first test run had been seen. The ranking used validation scenes only (`results/selection.json`), but this table is not a fully blind test of that choice.

All tables are generated by `scripts/make_report.py` from `results/*.json`.

## 1. PSNR and SSIM, 2. Colour accuracy (ΔE), 3. LPIPS

{metric_table([(v, summary_main[k]) for k, v in ARMS.items()], ARMS["hybrid_adaptive"])}

Higher is better for PSNR and SSIM. Lower is better for ΔE2000 and LPIPS.

Paired difference, deployed hybrid minus conventional baseline, bootstrapped over {summary["test"]["scenes"]} scenes:

| Metric | Hybrid minus traditional | Hybrid minus learned (int8) |
|---|---:|---:|
| PSNR (dB) | {fmt(deltas["psnr"], 2)} | {fmt(deltas_learned["psnr"], 2)} |
| SSIM | {fmt(deltas["ssim"], 4)} | {fmt(deltas_learned["ssim"], 4)} |
| ΔE2000 | {fmt(deltas["delta_e"], 2)} | {fmt(deltas_learned["delta_e"], 2)} |
| LPIPS | {fmt(deltas["lpips"], 4)} | {fmt(deltas_learned["lpips"], 4)} |

## 4. Model parameters and model size

{chr(10).join(model_lines)}

Classical stages have no learned parameters.

## 5. Inference latency and memory consumption

Measured on {machine["system"]}, {machine["cpu"]}, CPU only, with ONNX Runtime in a fresh process per measurement. Values are medians of 8 runs after one warm-up run. Peak memory is the working-set increase over the process baseline.

{chr(10).join(latency_lines)}

Classical stages, single process: {classical["256"]:.0f} ms at 256 px, {classical["512"]:.0f} ms at 512 px, {classical["1024"]:.0f} ms at 1024 px.

## Robustness under challenging conditions

Degradations are applied to the RAW mosaic with fixed seeds, so all arms receive identical inputs. Low-light rows pass the known digital gain to each pipeline, except the row marked as missing.

{chr(10).join(robust_lines)}

## Ablation study

Full analysis: [docs/ABLATION_AND_FAILURES.md](docs/ABLATION_AND_FAILURES.md).

Classical stage removal, clean test subset:

{metric_table([(r["label"], r) for r in summary["ablation"]["classical"]], "Full classical pipeline")}

Trained variants at equal width and iterations, clean test subset:

{metric_table([(r["label"], r) for r in summary["ablation"]["trained_clean"]], "Hybrid, all components")}

Trained variants, combined severe condition:

{metric_table([(r["label"], r) for r in summary["ablation"]["trained_severe"]], "Hybrid, all components")}
"""
    args.eval.write_text(text, encoding="utf-8")
    print("wrote", args.eval, "and", args.results / "summary.json")


if __name__ == "__main__":
    main()
