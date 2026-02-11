# Unified Attention U-Net for Cross-Modality Tumor Segmentation

This repository contains a modular PyTorch implementation for tumor segmentation across:
- BraTS 2021 (brain MRI)
- LIDC-IDRI (lung CT)

The project trains one unified Attention U-Net across both modalities.

## Authors
- Nishan Rai
- Pushparaj Dahal

## Datasets
- BraTS 2021: https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1
- LIDC-IDRI slices: https://www.kaggle.com/datasets/zhangweiled/lidcidri

## Requirements

Recommended environment:

```bash
conda create -n tumor_seg python=3.9
conda activate tumor_seg
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install numpy matplotlib albumentations nibabel pandas seaborn scikit-learn tqdm pillow
```

## Repository Structure

```text
.
├── src/
│   ├── data/                 # Datasets and transforms
│   ├── models/               # Attention U-Net model code
│   ├── training/             # Losses, samplers, training loops
│   └── eval/                 # Evaluation metrics and plots
├── scripts/
│   └── run_unified_training.py
├── artifacts/
│   ├── checkpoints/          # runtime .pth outputs (gitignored)
│   └── exports/              # runtime exports (e.g., ONNX, gitignored)
├── assets/
│   └── figures/              # architecture and flow images
└── data/
    └── samples/              # local sample/test artifacts
```

## Run Training

Use explicit dataset paths (no hardcoded machine-specific paths):

```bash
python3 scripts/run_unified_training.py \
  --brats-root /path/to/BraTS2021_Training_Data \
  --lidc-root /path/to/LIDC-IDRI-slices
```

Or with environment variables:

```bash
export BRATS_ROOT=/path/to/BraTS2021_Training_Data
export LIDC_ROOT=/path/to/LIDC-IDRI-slices
python3 scripts/run_unified_training.py
```

Optional args:
- `--num-patients` (default: `200`)
- `--seed` (default: `42`)
- `--epochs` (default: `45`)
- `--lr` (default: `1e-4`)

## Notes
- GPU/CUDA is recommended for training speed.
- `artifacts/` folders are kept for runtime outputs, but `.pth`/`.onnx` files are intentionally excluded from Git.


## Citation
If you use this project in research/publications, please cite:

> Rai, N., & Dahal, P. R. (2026). *A Unified Attention U-Net Framework for Cross-Modality Tumor Segmentation in MRI and CT*. arXiv:2601.06187. https://doi.org/10.48550/arXiv.2601.06187
