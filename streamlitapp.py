import streamlit as st
import zipfile
import tempfile
import os
import numpy as np
import torch
from PIL import Image
import nibabel as nib
import matplotlib.pyplot as plt

# Define model architecture
class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class AttentionBlock(torch.nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = torch.nn.Sequential(
            torch.nn.Conv2d(F_g, F_int, 1),
            torch.nn.BatchNorm2d(F_int)
        )
        self.W_x = torch.nn.Sequential(
            torch.nn.Conv2d(F_l, F_int, 1),
            torch.nn.BatchNorm2d(F_int)
        )
        self.psi = torch.nn.Sequential(
            torch.nn.Conv2d(F_int, 1, 1),
            torch.nn.BatchNorm2d(1),
            torch.nn.Sigmoid()
        )
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(torch.nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        features = [64, 128, 256]
        self.encoder1 = DoubleConv(in_channels, features[0])
        self.pool1 = torch.nn.MaxPool2d(2)
        self.encoder2 = DoubleConv(features[0], features[1])
        self.pool2 = torch.nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(features[1], features[2])
        self.up2 = torch.nn.ConvTranspose2d(features[2], features[1], 2, stride=2)
        self.att2 = AttentionBlock(features[1], features[1], features[1]//2)
        self.decoder2 = DoubleConv(features[2], features[1])
        self.up1 = torch.nn.ConvTranspose2d(features[1], features[0], 2, stride=2)
        self.att1 = AttentionBlock(features[0], features[0], features[0]//2)
        self.decoder1 = DoubleConv(features[1], features[0])
        self.final_conv = torch.nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        a2 = self.att2(d2, e2)
        d2 = self.decoder2(torch.cat([d2, a2], 1))
        d1 = self.up1(d2)
        a1 = self.att1(d1, e1)
        d1 = self.decoder1(torch.cat([d1, a1], 1))
        return self.final_conv(d1)

@st.cache_resource
def load_model():
    model = AttentionUNet()
    model.load_state_dict(torch.load("best_model.pth", map_location='cpu'))
    model.eval()
    return model

def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

def process_zip(uploaded_zip):
    temp_dir = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir.name)
    
    extracted_files = []
    for root, _, files in os.walk(temp_dir.name):
        for file in files:
            extracted_files.append(os.path.join(root, file))
    
    is_nifti = any(f.endswith((".nii.gz", ".nii")) for f in extracted_files)
    is_png = any(f.endswith(".png") for f in extracted_files)
    
    volume = []
    mask = None
    
    if is_nifti:
        modalities = {"t1": None, "t1ce": None, "t2": None, "flair": None}
        for f in extracted_files:
            fname = os.path.basename(f).lower()
            for mod in modalities:
                if fname.endswith(f"_{mod}.nii.gz") or fname.endswith(f"_{mod}.nii"):
                    modalities[mod] = f
            if "seg" in fname:
                mask = f
        
        for mod in modalities.values():
            if not mod:
                st.error(f"Missing {mod} modality in NIfTI files")
                return None, None, temp_dir
        
        for mod in ["t1", "t1ce", "t2", "flair"]:
            img = nib.load(modalities[mod]).get_fdata()
            slice_idx = img.shape[-1] // 2
            img_slice = normalize(img[:, :, slice_idx])
            img_slice = Image.fromarray(img_slice).resize((128, 128))
            volume.append(np.array(img_slice))
        
        if mask:
            mask_data = nib.load(mask).get_fdata()
            mask_slice = mask_data[:, :, mask_data.shape[-1] // 2]
            mask = Image.fromarray(mask_slice).resize((128, 128))
            mask = np.array(mask) > 0
    
    elif is_png:
        png_files = {"t1": None, "t1ce": None, "t2": None, "flair": None, "mask": None}
        for f in extracted_files:
            fname = os.path.basename(f).lower()
            for mod in png_files:
                if fname.endswith(f"_{mod}.png") or fname == f"{mod}.png":
                    png_files[mod] = f
        
        for mod in ["t1", "t1ce", "t2", "flair"]:
            if not png_files[mod]:
                st.error(f"Missing {mod}.png")
                return None, None, temp_dir
            
            img = Image.open(png_files[mod]).convert('L').resize((128, 128))
            volume.append(normalize(np.array(img)))
        
        if png_files["mask"]:
            mask = Image.open(png_files["mask"]).convert('L').resize((128, 128))
            mask = np.array(mask) > 0
    
    return np.stack(volume), mask, temp_dir

# Streamlit UI
st.title("Brain Tumor Segmentation App")
st.write("Upload a ZIP file containing medical imaging data")

uploaded_file = st.file_uploader("Choose a ZIP file", type=["zip"])

if uploaded_file:
    volume, mask, temp_dir = process_zip(uploaded_file)
    
    if volume is not None:
        input_tensor = torch.tensor(volume).unsqueeze(0).float()
        model = load_model()
        
        with torch.no_grad():
            pred = torch.sigmoid(model(input_tensor))
            pred_mask = (pred.squeeze().numpy() > 0.5).astype(np.uint8)
        
        fig, axes = plt.subplots(1, 3 if mask is not None else 2, figsize=(15, 5))
        axes[0].imshow(volume[-1], cmap='gray')
        axes[0].set_title("Input Image (FLAIR)")
        axes[0].axis('off')
        
        if mask is not None:
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title("Ground Truth")
            axes[1].axis('off')
            axes[2].imshow(pred_mask, cmap='gray')
            axes[2].set_title("Prediction")
            axes[2].axis('off')
        else:
            axes[1].imshow(pred_mask, cmap='gray')
            axes[1].set_title("Prediction")
            axes[1].axis('off')
        
        st.pyplot(fig)
    
    if temp_dir:
        temp_dir.cleanup()