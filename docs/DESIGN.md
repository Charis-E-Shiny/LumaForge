# Design Document

> **Status: complete.** This records the design, the evidence behind it, and where the evidence overturned the original plan. Section 2.7 and Section 6 describe the deployed system.

## 1. Goal

Convert degraded smartphone RAW into high-quality sRGB, while improving colour accuracy, noise reduction and sharpness under mobile compute limits.

## 2. Key Decisions

### 2.1 Residual refinement instead of end-to-end rendering

**Decision:** The compact network refines an undenoised classical render.

**Why:** The classical stages encode sensor physics and cost no parameters, and colour stays anchored to a real render. The original design refined a fully denoised render. The ablation study showed that this removes texture the network cannot restore, so the classical denoiser was removed from the compact model's base. See Section 2.7.

### 2.2 One NumPy implementation of the classical stages

**Decision:** No OpenCV or Torch in the classical branch.

**Why:** The same code runs in the data loader and the serverless function, which rules out train/serve skew and keeps the deployment small. The cost is some speed compared with optimised native code.

### 2.3 Noise measured from the mosaic

**Decision:** Fit $\sigma^2 = aI + b$ per capture from the two green planes.

**Why:** Test blocks carry no capture metadata, and a blind measurement works for any input. The green-plane difference cancels most image structure.

### 2.4 Known digital gain, not auto-exposure

**Decision:** Exposure compensation is an input.

**Why:** On training crops, linear signal maps to the target about 1:1, so references keep dark scenes dark. Blind auto-exposure would brighten those scenes and lose fidelity. Real phones know their shutter and ISO, so the input is realistic. The missing-metadata case is scheduled as a failure test.

### 2.5 Fitted colour matrices

**Decision:** Fit one matrix per camera by least squares on training scenes.

**Why:** Matrices derived from DNG `ColorMatrix2` showed a consistent bias of about +19% in the red ratio and -11% in the blue ratio across cameras.

### 2.6 White balance with a camera prior

**Decision:** Shrink shades-of-grey estimates towards a per-camera prior in log space, with weight 0.8.

**Why:** White balance is the largest classical error. On sampled training crops, as-shot white balance gave 37.1 dB and ΔE76 3.3, and plain shades-of-grey gave 30.2 dB and ΔE76 9.5. The prior reduced mean illuminant error from 7.3° to 4.2°. Crops are small, so one object can dominate the estimate.

**Caveat:** The prior comes from the dataset's illuminants, so it may not transfer to other cameras or lighting.

### 2.7 Two routed models, chosen from evidence

**Decision:** A compact hybrid (width 12, int8) handles low-noise captures. The learned model (width 24, int8) handles captures above a post-gain noise gate of `0.0284`.

**Why:** The first deployed pair, a distilled student and teacher on a denoised base, barely beat the classical baseline on the test set. Checkpoints were then ranked on validation scenes across PSNR, SSIM, CIEDE2000 and LPIPS (`scripts/select_model.py`). The learned model ranked first, and the hybrid without classical denoising ranked second, 0.12 dB behind at about 40% of the file size.

**Deviation from the ranking rule:** The rule named one winner. The compact hybrid was kept as the default route anyway, because the brief asks for an efficient hybrid. The ranking winner serves the quality route.

**Gate:** `scripts/tune_gate.py` picks the gate on validation scenes: the largest fast-route share with mean PSNR within 0.05 dB of always using the quality model. This gave `0.0284` and a 70% fast share.

**Caveat:** This pair was chosen after the first test run had been seen. The ranking used validation data only, but the published test numbers are not a fully blind test of the choice.

**Consequence:** The deployed system does not use distillation or fp16. Both remain implemented and measured in `eval_P2_T32.md`.

### 2.8 Quantization boundaries

**Decision:** The output head and global colour head stay in floating point.

**Why:** The output adds directly to the render, so a coarsely quantized residual would appear as banding.

### 2.9 MongoDB is optional

**Decision:** Database calls have short timeouts and never block processing.

**Why:** Telemetry must not reduce availability of the core feature.

## 3. Data

See [DATASETS.md](DATASETS.md). The official SIDD server was unreachable, so the project uses SIDD-CC.

Known effects:

- Training crops are 256 × 256, so the network sees little global context for colour.
- One capture per scene is used, which limits disk use on the training machine.
- Tuning for white balance sampled cached scenes without excluding validation scenes. This affects validation slightly. It does not touch the test split.

## 4. Training

| Run | Width | Iterations | Notes |
|---|---:|---:|---|
| Teacher | 24 | 5,000 | Hybrid on a denoised base, not deployed |
| Learned | 24 | 5,000 | Deployed on the quality route |
| Student | 12 | 4,000 | Distilled from the teacher, not deployed |
| Ablation variants | 12 | 2,000 | One change each. `abl_no_classical_denoise` is deployed as the compact hybrid |

These budgets reflect CPU-only training on a laptop. The whole sequence took about 7.5 hours.

## 5. Risks

| Risk | Mitigation |
|---|---|
| Small crops limit global colour correction | Report per-camera results and analyse failures |
| Quantization harms quality | Compare fp32 and int8 on the same blocks |
| Short CPU training underfits | Keep the classical base, so the starting point is already reasonable |
| Results depend on one dataset | State this limitation, and do not generalise beyond it |

## 6. Results

Measured on 928 held-out blocks from 29 scenes:

| Pipeline | PSNR (dB) | SSIM | ΔE2000 | LPIPS | Model size |
|---|---:|---:|---:|---:|---:|
| Traditional baseline | 30.90 | 0.8557 | 5.53 | 0.3287 | none |
| Learned, int8 | 30.97 | 0.8838 | 5.82 | 0.2230 | 0.73 MB |
| Compact hybrid, int8 | 31.01 | 0.8823 | 5.47 | 0.2124 | 0.29 MB |
| **Hybrid adaptive (deployed)** | **30.92** | **0.8832** | **5.61** | **0.2170** | **0.29 + 0.73 MB** |

Against the baseline, the deployed system changes SSIM by +0.0276 (95% CI +0.0167 to +0.0395) and LPIPS by -0.1117 (95% CI -0.1436 to -0.0832). PSNR changes by +0.02 dB (95% CI -0.58 to +0.62) and ΔE2000 by +0.08 (95% CI -0.32 to +0.43), so neither differs reliably from the baseline.

The system clearly improves structure and perceived quality. It does not improve mean fidelity or colour accuracy on clean captures. Its largest fidelity gains are under stress, such as +1.05 dB at 3 stops under.

On test, the compact model alone slightly beat the routed system, so routing did not earn its cost on clean captures. See [ABLATION_AND_FAILURES.md](ABLATION_AND_FAILURES.md).

Full tables are in [`eval_P2_T32.md`](../eval_P2_T32.md).
