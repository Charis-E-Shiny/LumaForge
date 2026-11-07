# Changelog

Notable changes to this project are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Classical NumPy ISP covering demosaicing, white balance, colour correction, denoising, tone mapping, sharpening and colour-space transforms
- Blind noise fitting and an adaptive control plan
- LumaNet in hybrid and learned forms
- RAW-domain degradations for training and robustness testing
- SIDD-CC data preparation, calibration and tuning scripts
- Training, distillation, ONNX export, quantization, evaluation, benchmark and reporting scripts
- FastAPI function with optional MongoDB telemetry
- Next.js interface with Lab, Results and Architecture pages
- Tests and GitHub Actions CI
- Project documentation and community files
- Trained models, evaluation results, `eval_P2_T32.md`, and the ablation and failure analysis
- Validation model ranking and noise-gate tuning scripts

### Changed

- Replaced blind auto-exposure with known digital gain, because SIDD targets keep dark scenes dark
- Replaced DNG-derived colour matrices with matrices fitted on training scenes
- Added a per-camera prior to white-balance estimation

### Fixed

- Corrected the sign of a coefficient in the Malvar demosaicing filters

### Pending

- Trained models, evaluation results, ablation and failure analysis, and the live deployment
