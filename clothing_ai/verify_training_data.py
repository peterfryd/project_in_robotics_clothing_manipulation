import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# ==== CONFIG (Must match finetune.py exactly) ====
NEW_ANNOS_DIR = "./data/annos"
NEW_IMAGES_DIR = "./data/images"
IMG_SIZE = 224

# ==== REPLICATE YOUR DATASET LOGIC EXACTLY ====
class DebugDataset(Dataset):
    def __init__(self, annos_dir, images_dir, transform=None):
        self.transform = transform
        self.samples = []
        for jf in sorted([f for f in os.listdir(annos_dir) if f.endswith('.json')]):
            with open(os.path.join(annos_dir, jf), 'r') as f:
                data = json.load(f)
            self.samples.append({
                'img_path': os.path.join(images_dir, data['image']),
                'landmarks': data['landmarks']
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item['img_path']).convert("RGB")
        
        # --- CRITICAL SECTION: Coordinate Normalization ---
        # Are we SURE it is 0-100 percentage? Test it here.
        landmarks = torch.tensor(item['landmarks'], dtype=torch.float32)
        
        # TEST 1: Assume standard Label Studio 0-100 percentage
        landmarks[:, 0] /= 100.0
        landmarks[:, 1] /= 100.0
        
        # TEST 2: (Uncomment if TEST 1 fails) Assume raw pixels
        # w, h = img.size
        # landmarks[:, 0] /= w
        # landmarks[:, 1] /= h
        # --------------------------------------------------

        if self.transform:
            img = self.transform(img)

        return img, landmarks

# ==== VISUALIZATION ====
def show_batch(images, landmarks):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8)) # Show 8 images
    axes = axes.flatten()
    
    for i in range(min(len(images), 8)):
        ax = axes[i]
        # Un-normalize image for display
        img_np = images[i].permute(1, 2, 0).numpy()
        # Clamp just in case color jitter pushed values outside [0,1]
        img_np = np.clip(img_np, 0, 1)
        
        ax.imshow(img_np)
        ax.axis('off')
        ax.set_title(f"Sample {i}")
        
        # Landmarks are now normalized [0, 1] relative to the 224x224 tensor
        lms = landmarks[i]
        for (x, y, v) in lms:
            if v > 0.1: # Only show visible
                # Scale back to 224 for visualization on the tensor
                gx = x * IMG_SIZE
                gy = y * IMG_SIZE
                circ = patches.Circle((gx, gy), radius=4, color='lime')
                ax.add_patch(circ)

    plt.tight_layout()
    plt.show()

# ==== RUN CHECK ====
if __name__ == '__main__':
    # Use RESIZE only, no jitter, to see clearly
    debug_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    try:
        ds = DebugDataset(NEW_ANNOS_DIR, NEW_IMAGES_DIR, transform=debug_tfms)
        dl = DataLoader(ds, batch_size=8, shuffle=True)
        
        print("🔎 Fetching one batch from DataLoader...")
        imgs, lms = next(iter(dl))
        print(f"✅ Loaded batch. Images shape: {imgs.shape}, Landmarks shape: {lms.shape}")
        print("Check the popup window. Do the LIME GREEN dots match the landmarks exactly?")
        show_batch(imgs, lms)
    except Exception as e:
        print(f"❌ Error during data check: {e}")