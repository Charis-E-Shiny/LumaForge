# Datasets

## SIDD-CC (Used)

[SIDD-CC](https://huggingface.co/datasets/MhdMohammadi/SIDD-CC) is a re-processing of the Smartphone Image Denoising Dataset, with re-estimated ground truth and a new split.

| Property | Value |
|---|---|
| Format | Bayer `float32`, black-level subtracted and unclipped; 8-bit sRGB |
| Train | 115 instances, full resolution |
| Test | 29 instances, 32 blocks of 256 × 256 each, 928 in total |
| Cameras | Galaxy S6, Google Pixel, Nexus 6, LG G4, iPhone 7 |
| Test metadata | Not provided, so white balance must be estimated |

**Why not the official SIDD files?** The server listed on the SIDD site was unreachable during this work, and its mirror list uses links that need a login.

## Local Preparation

`scripts/prepare_sidd_cc.py` streams one capture per training instance, keeps 24 crops of 256 × 256, and discards the originals.

| Step | Detail |
|---|---|
| Inputs | Noisy RAW and ground-truth sRGB |
| Crop phase | Every crop starts on a red sample, so all cameras share RGGB |
| Storage | RAW as `int16`, which keeps negative sub-black noise |
| Seeds | Derived from instance names, so crops are reproducible |

`scripts/consolidate_cache.py` then produces:

| Split | Scenes | Crops |
|---|---:|---:|
| Train | 103 | 2,472 |
| Validation | 12 | 288 |

## Test Split Camera Mapping

Test blocks do not name their cameras. Each camera is identified from the full-frame size in `test_scenes_info.txt`, and these counts match the dataset card:

| Frame size | Camera | Scenes |
|---|---|---:|
| 3036 × 4048 | Google Pixel | 9 |
| 2988 × 5312 | LG G4 | 7 |
| 3120 × 4208 | Nexus 6 | 6 |
| 3024 × 4032 | iPhone 7 | 6 |
| 3000 × 5328 | Galaxy S6 | 1 |

The test coordinates are all even, so each block keeps its camera's CFA phase.

## Modular Neural ISP

[The Samsung Labs repository](https://github.com/SamsungLabs/modular_neural_isp) contains code and dataset-preparation scripts, and no image data. Its main experiments use the S24 Raw-sRGB dataset, which is distributed through sync.com. S24 is not used here. The repository informs the modular design, and none of its code is included.

## Terms

Read the [SIDD terms](https://abdokamel.github.io/sidd/) before reusing data. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
