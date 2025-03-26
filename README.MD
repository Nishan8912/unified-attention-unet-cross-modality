# Tumor Segmentation with U-Net

This repository contains code for tumor segmentation tasks using a simplified U-Net architecture implemented in PyTorch.

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
- Adjust hyperparameters and dataset paths as necessary for your data and task.



