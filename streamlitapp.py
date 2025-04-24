import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import tempfile
import shutil
from PIL import Image
import nibabel as nib
import matplotlib.pyplot as plt

# -----------------------------
# Model Definitions
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        return x * self.psi(psi)

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        features = [64, 128, 256]
        self.encoder1 = DoubleConv(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = DoubleConv(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(features[1], features[2])
        self.up2 = nn.ConvTranspose2d(features[2], features[1], 2, stride=2)
        self.att2 = AttentionBlock(features[1], features[1], features[1] // 2)
        self.decoder2 = DoubleConv(features[2], features[1])
        self.up1 = nn.ConvTranspose2d(features[1], features[0], 2, stride=2)
        self.att1 = AttentionBlock(features[0], features[0], features[0] // 2)
        self.decoder1 = DoubleConv(features[1], features[0])
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        a2 = self.att2(d2, e2)
        d2 = self.decoder2(torch.cat([d2, a2], dim=1))
        d1 = self.up1(d2)
        a1 = self.att1(d1, e1)
        d1 = self.decoder1(torch.cat([d1, a1], dim=1))
        return self.final_conv(d1)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = AttentionUNet()
    model.load_state_dict(torch.load("best_model.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()

# -----------------------------
# Helper functions
# -----------------------------
def normalize(img):
    img = img.astype(np.float32)
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

def process_uploaded_folder(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
    return temp_dir

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Tumor Segmentation (BraTS / LIDC Folder Upload)")

uploaded_folder = st.file_uploader("Upload all files from one test folder (e.g. sample0)", type=["nii.gz", "png"], accept_multiple_files=True)

if uploaded_folder:
    folder_path = process_uploaded_folder(uploaded_folder)
    files = os.listdir(folder_path)
    modality_names = ["t1", "t1ce", "t2", "flair"]

    is_nifti = any(f.endswith(".nii.gz") for f in files)
    is_png = any(f.endswith(".png") for f in files)

    volume = []
    if is_nifti:
        st.success("Detected format: BraTS (NIfTI)")
        sample_id = files[0].split("_")[0]
        for name in modality_names:
            path = os.path.join(folder_path, f"{sample_id}_{name}.nii.gz")
            img = nib.load(path).get_fdata()
            if img.ndim == 3:
                slice_ = normalize(img[:, :, img.shape[-1] // 2])
            elif img.ndim == 2:
                slice_ = normalize(img)
            else:
                st.error(f"Unsupported image shape: {img.shape}")
                st.stop()
            volume.append(slice_)

        mask_path = os.path.join(folder_path, f"{sample_id}_seg.nii.gz")
        mask_data = nib.load(mask_path).get_fdata()
        mask = (mask_data[:, :, mask_data.shape[-1] // 2] > 0).astype(np.uint8)

    elif is_png:
        st.success("Detected format: LIDC-IDRI (PNG)")
        for name in modality_names:
            path = os.path.join(folder_path, f"{name}.png")
            img = Image.open(path).convert("L").resize((128, 128))
            volume.append(normalize(np.array(img)))

        mask = Image.open(os.path.join(folder_path, "mask.png")).convert("L").resize((128, 128))
        mask = (np.array(mask) > 0).astype(np.uint8)

    else:
        st.error("Could not determine format. Upload .nii.gz or .png files.")
        st.stop()

    # Predict
    input_tensor = torch.tensor(np.stack(volume)).unsqueeze(0).float()
    with torch.no_grad():
        pred = torch.sigmoid(model(input_tensor))
        pred_mask = (pred.squeeze().numpy() > 0.5).astype(np.uint8)

    # Display
    st.subheader("Input & Prediction")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(volume[-1], cmap="gray")
    plt.title("Input Modality")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground Truth")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Predicted Mask")

    st.pyplot(plt)
    shutil.rmtree(folder_path)
else:
    st.info("Upload all test files from a sample folder (.nii.gz for BraTS, .png for LIDC)")
