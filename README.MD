Multi-Modal Deep Learning for Tumor Segmentation
-This repository contains code for tumor segmentation 
tasks using a simplified U-Net architecture 
implemented in PyTorch.
===============================================

Team Members:
- Nishan Rai
- Pushparaj Dahal

Dataset Information
-----------------------------------------------------

**1. BraTS 2021 Dataset (Brain MRI Tumor Segmentation)**:
- Source: Kaggle
- Link: https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1
- Format: .nii.gz volumes with 4 modalities (T1, T1ce, T2, FLAIR)
- Used for: Brain tumor segmentation

**2. LIDC-IDRI Dataset (Lung CT Nodule Segmentation)**:
- Source: Kaggle
- Link: https://www.kaggle.com/datasets/zhangweiled/lidcidri
- Format: Pre-sliced CT scans with 4 radiologist annotations per nodule
- Used for: Lung nodule segmentation

## Dataset Organization

Download the datasets and organize as follows:

├── datasets/
│   ├── BraTS/
│   │   └── [Extracted .nii.gz files]
│   └── LIDC/
│       └── [CT slices and segmentation masks]

Update the paths in the notebook accordingly.

## Requirements

Install the required Python packages using pip:

```bash
pip install torch torchvision numpy matplotlib segmentation-models-pytorch notebook
```

Alternatively, create and activate a conda environment:

```bash
conda create -n tumor_seg python=3.9
conda activate tumor_seg
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install numpy matplotlib segmentation-models-pytorch notebook
```

## Key Dependencies

- **torch** (PyTorch)
- **torchvision**
- **segmentation-models-pytorch**
- **numpy**
- **matplotlib** (for visualization)

## Running the Notebook

1. Open the terminal and navigate to the directory containing the notebook (`tumor_seg.ipynb`).

2. Launch Jupyter Notebook:

```bash
jupyter notebook tumor_seg.ipynb
```

3. Run the notebook cells sequentially to train, evaluate, and visualize your segmentation model.

## File Structure

- `tumor_seg.ipynb`: Main notebook containing model definition, training, and evaluation code.

> **Important:** Make sure your dataset paths are correctly set inside the notebook.

## Notes

- Ensure CUDA and GPU compatibility if using GPU acceleration.
- Adjust dataset paths as necessary for your data and task.

## Submission Structure

This project submission includes:
- report.pdf: Final report
- team.txt: Team member list
- readme.md: This file
- tumor_seg.ipynb: Code notebook
