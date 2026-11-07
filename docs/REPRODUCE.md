# Reproducing the Results

Run all commands from the repository root with the virtual environment active.

> The training sequence is planned at about five hours on a 4-core laptop CPU. That estimate comes from short speed tests, not from a completed run.

## 1. Test Data

Download these files into `data/sidd_cc/test/` from [SIDD-CC](https://huggingface.co/datasets/MhdMohammadi/SIDD-CC):

| Source path | Save as |
|---|---|
| `Raw/test/BenchmarkNoisyBlocksRaw.mat` | `raw_noisy.mat` |
| `Raw/test/BenchmarkGtBlocksRaw.mat` | `raw_gt.mat` |
| `SRGB/test/BenchmarkGtBlocksSrgb.mat` | `srgb_gt.mat` |
| `Raw/test/BenchmarkBlocks32.mat` | `BenchmarkBlocks32.mat` |
| `Raw/test/test_scenes_info.txt` | `test_scenes_info.txt` |

## 2. Training Data and Calibration

```bash
python scripts/prepare_sidd_cc.py --out data/cache --crops 24 --size 256 --workers 3
python scripts/consolidate_cache.py --cache data/cache --out data/crops
python scripts/calibrate.py --cache data/cache
python scripts/tune_classical.py --cache data/cache --write
```

## 3. Training

```bash
sh scripts/run_training.sh
```

## 4. Export and Samples

```bash
python scripts/export_onnx.py
python scripts/make_samples.py
```

## 5. Evaluation

```bash
python scripts/select_model.py
python scripts/tune_gate.py
python scripts/evaluate.py main
python scripts/evaluate.py robust
python scripts/evaluate.py ablation
python scripts/benchmark.py
python scripts/make_report.py
python scripts/make_figures.py
```

`make_report.py` writes `eval_P2_T32.md` and `results/summary.json`.
