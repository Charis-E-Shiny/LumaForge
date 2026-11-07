# Third-Party Notices

The [MIT License](LICENSE) covers LumaForge source code only. The items below keep their own terms.

## Datasets

| Dataset | Terms | How it is used |
|---|---|---|
| [SIDD](https://abdokamel.github.io/sidd/) | [SIDD terms of use](https://abdokamel.github.io/sidd/) | Origin of SIDD-CC |
| [SIDD-CC](https://huggingface.co/datasets/MhdMohammadi/SIDD-CC) | SIDD terms of use, as stated by its dataset card | Training, validation, testing and demo captures |

Trained models and demo captures are derived from this data, so the MIT License does not extend to them. Read the dataset terms before reusing them, particularly for commercial purposes.

## Referenced Work

[Modular Neural ISP](https://github.com/SamsungLabs/modular_neural_isp) by Samsung Labs is licensed under CC BY-NC-SA 4.0. LumaForge uses it as a design reference. No code from it is included in this repository.

## Software Dependencies

Dependencies are installed from their own sources and keep their own licenses. They include PyTorch, ONNX Runtime, NumPy, SciPy, scikit-image, LPIPS, FastAPI, PyMongo, Next.js and React. See [`requirements.txt`](requirements.txt), [`requirements-train.txt`](requirements-train.txt) and [`package.json`](package.json).

## Published Methods

The classical stages are original implementations of published methods:

- H. S. Malvar, L. He and R. Cutler, "High-quality linear interpolation for demosaicing of Bayer-patterned color images", ICASSP 2004.
- K. He, J. Sun and X. Tang, "Guided Image Filtering", IEEE TPAMI 2013.
- T. Brooks et al., "Unprocessing Images for Learned Raw Denoising", CVPR 2019.
