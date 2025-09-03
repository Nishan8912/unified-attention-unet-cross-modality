# Tumor Segmentation using Attention UNet

This project implements an Attention UNet architecture for brain tumor segmentation from MRI scans. The model combines the power of UNet with attention mechanisms to improve segmentation accuracy.

## Model Architecture

### Attention UNet Overview

The Attention UNet is an enhanced version of the traditional UNet that incorporates attention gates to focus on relevant features during the decoding process. This architecture is particularly effective for medical image segmentation tasks.

### Architecture Block Diagram

```mermaid
graph TB
    %% Input Layer
    Input[Input: 4-channel MRI] --> Encoder1
    
    %% Top Row: Encoder Path (Left to Right)
    subgraph Encoder["Encoder Path"]
        Encoder1[Encoder1: DoubleConv<br/>4 → 64 channels] --> Pool1[MaxPool2D<br/>128×128 → 64×64]
        Pool1 --> Encoder2[Encoder2: DoubleConv<br/>64 → 128 channels] --> Pool2[MaxPool2D<br/>64×64 → 32×32]
        Pool2 --> Bottleneck[Bottleneck: DoubleConv<br/>128 → 256 channels]
    end
    
    %% Bottleneck with Dropout (Center)
    Bottleneck --> Dropout[Dropout2D<br/>p=0.3]
    
    %% Bottom Row: Decoder Path (Right to Left)
    subgraph Decoder["Decoder Path"]
        Up2[ConvTranspose2D<br/>256 → 128 channels<br/>32×32 → 64×64] --> Att2[Attention Gate 2<br/>128 → 64 channels]
        Att2 --> Decoder2[Decoder2: DoubleConv<br/>256 → 128 channels]
        Decoder2 --> Up1[ConvTranspose2D<br/>128 → 64 channels<br/>64×64 → 128×128]
        Up1 --> Att1[Attention Gate 1<br/>64 → 32 channels]
        Att1 --> Decoder1[Decoder1: DoubleConv<br/>128 → 64 channels]
        Decoder1 --> FinalConv[Final Conv<br/>64 → 1 channel]
        FinalConv --> Output[Output: Segmentation Mask]
    end
    
    %% Connect Bottleneck to Decoder
    Dropout --> Up2
    
    %% Skip Connections (Vertical)
    Encoder2 -.-> Att2
    Encoder1 -.-> Att1
    
    %% Styling
    classDef encoder fill:#e1f5fe
    classDef decoder fill:#f3e5f5
    classDef attention fill:#fff3e0
    classDef bottleneck fill:#e8f5e8
    
    class Encoder1,Encoder2 encoder
    class Decoder1,Decoder2 decoder
    class Att1,Att2 attention
    class Bottleneck bottleneck
```

### Detailed Architecture Components

#### 1. Encoder Path (Contracting)
- **Encoder1**: DoubleConv block (4 → 64 channels)
  - Conv2D(4, 64, kernel=3×3, padding=1) + BatchNorm2D + ReLU
  - Conv2D(64, 64, kernel=3×3, padding=1) + BatchNorm2D + ReLU
  - MaxPool2D(2×2)

- **Encoder2**: DoubleConv block (64 → 128 channels)
  - Conv2D(64, 128, kernel=3×3, padding=1) + BatchNorm2D + ReLU
  - Conv2D(128, 128, kernel=3×3, padding=1) + BatchNorm2D + ReLU
  - MaxPool2D(2×2)

#### 2. Bottleneck
- **Bottleneck**: DoubleConv block (128 → 256 channels)
  - Conv2D(128, 256, kernel=3×3, padding=1) + BatchNorm2D + ReLU
  - Conv2D(256, 256, kernel=3×3, padding=1) + BatchNorm2D + ReLU
  - **Dropout2D(p=0.3)** for regularization

#### 3. Decoder Path (Expanding) with Attention Gates
- **Up2**: ConvTranspose2D(256, 128, kernel=2×2, stride=2)
- **Attention Gate 2**: 
  - Processes gating signal (128 channels) and local features (128 channels)
  - Outputs attention weights (64 channels)
- **Decoder2**: DoubleConv block (256 → 128 channels)
  - Concatenates upsampled features with attention-weighted encoder features

- **Up1**: ConvTranspose2D(128, 64, kernel=2×2, stride=2)
- **Attention Gate 1**:
  - Processes gating signal (64 channels) and local features (64 channels)
  - Outputs attention weights (32 channels)
- **Decoder1**: DoubleConv block (128 → 64 channels)
  - Concatenates upsampled features with attention-weighted encoder features

#### 4. Output Layer
- **Final Conv**: Conv2D(64, 1, kernel=1×1)
- **Output**: Single-channel segmentation mask

### Attention Mechanism

The attention gates use the following components:
- **W_g**: 1×1 convolution to process gating signal
- **W_x**: 1×1 convolution to process local features
- **ψ**: 1×1 convolution to generate attention weights
- **Sigmoid activation** to normalize attention weights

### Model Parameters

- **Input Channels**: 4 (multi-modal MRI: T1, T1c, T2, FLAIR)
- **Output Channels**: 1 (binary segmentation mask)
- **Feature Dimensions**: [64, 128, 256]
- **Total Parameters**: ~1.2M
- **Input Size**: 128×128 pixels
- **Output Size**: 128×128 pixels

### Key Features

1. **Attention Gates**: Focus on relevant features during decoding
2. **Skip Connections**: Preserve spatial information from encoder
3. **Double Convolutions**: Enhanced feature extraction
4. **Dropout Regularization**: Prevents overfitting
5. **Batch Normalization**: Stabilizes training

### Training Details

- **Loss Function**: Combined Dice Loss + Focal Loss
- **Optimizer**: Adam with learning rate 3e-5
- **Data Augmentation**: Albumentations library
- **Batch Size**: 8
- **Epochs**: 30

### Dataset

The model is trained on:
- **BraTS 2020**: Brain tumor segmentation challenge dataset
- **LIDC-IDRI**: Lung nodule dataset for transfer learning

This architecture demonstrates state-of-the-art performance in medical image segmentation tasks, particularly for brain tumor detection and segmentation.
