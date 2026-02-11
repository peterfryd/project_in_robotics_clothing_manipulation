import os
import torch
import numpy as np
import pandas as pd
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import glob
import json
from tqdm import tqdm

# ==== CONFIGURATION ==========================================================
DATA_DIR = "clothing_ai/data/deepFashion2/validation"
CKPT_PATH = "clothing_ai/checkpoints/model.pth"
OUTPUT_CSV = "validation_complete_metrics.csv"

IMG_SIZE = 224
NUM_LANDMARKS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== MODEL DEFINITION =======================================================
class LandmarkRegressor(nn.Module):
    def __init__(self, num_landmarks):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_landmarks * 3)

    def forward(self, x):
        return self.backbone(x).view(-1, self.num_landmarks, 3)

# ==== INITIALIZATION =========================================================
model = LandmarkRegressor(NUM_LANDMARKS).to(DEVICE)
checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

image_files = glob.glob(os.path.join(DATA_DIR, "images", "*.jpg")) + \
              glob.glob(os.path.join(DATA_DIR, "images", "*.png"))

# Headers: ID, GT(x,y,v), Pred(x,y in px), Dist(px), NME
headers = ['image_id', 'orig_w', 'orig_h', 'diagonal']
for i in range(1, NUM_LANDMARKS + 1):
    headers.extend([f'L{i}_gt_x', f'L{i}_gt_y', f'L{i}_gt_v', 
                    f'L{i}_pred_px_x', f'L{i}_pred_px_y', 
                    f'L{i}_dist_px', f'L{i}_nme'])

# ==== PROCESSING =============================================================
data_rows = []

for img_path in tqdm(image_files, desc="Running Inference"):
    image_id = os.path.splitext(os.path.basename(img_path))[0]
    ann_path = os.path.join(DATA_DIR, "annotations", f"{image_id}.json")
    
    if not os.path.exists(ann_path): continue

    try:
        # Load GT and Image Info
        with open(ann_path, "r") as f:
            anno = json.load(f)
        gt_landmarks = np.array(anno["landmarks"], dtype=np.float32)

        orig_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = orig_pil.size
        diagonal = np.sqrt(orig_w**2 + orig_h**2)
        
        inp = transform(orig_pil).unsqueeze(0).to(DEVICE)
        orig_pil.close()

        with torch.no_grad():
            preds_norm = model(inp).cpu().numpy().reshape(NUM_LANDMARKS, 3)
        
        row = [image_id, orig_w, orig_h, diagonal]
        
        for i in range(NUM_LANDMARKS):
            # 1. GT Info
            gx, gy, gv = gt_landmarks[i]
            
            # 2. Predicted Pixels (Scaled from 0-1 to Orig Image Size)
            px = preds_norm[i, 0] * orig_w
            py = preds_norm[i, 1] * orig_h
            
            # 3. Calculate metrics
            dist_px = np.sqrt((px - gx)**2 + (py - gy)**2)
            nme = dist_px / diagonal
            
            row.extend([gx, gy, gv, px, py, dist_px, nme])
            
        data_rows.append(row)

    except Exception as e:
        print(f"Error on {image_id}: {e}")

df = pd.DataFrame(data_rows, columns=headers)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ CSV Saved: {OUTPUT_CSV}")