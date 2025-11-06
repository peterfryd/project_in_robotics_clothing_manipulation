import os
import cv2
import json
import torch
import numpy as np
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# ==== CONFIG ====
IMG_PATH = "/home/peter/uni/clothing_ai/Data/validation/images/000555.jpg"
IMG_PATH = "/home/peter/uni/clothing_ai/4.jpg"
ANN_PATH = "/home/peter/uni/clothing_ai/Data/validation/annotations/000555.json"
CKPT_PATH = "./checkpoints/model_step_15000.pth"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== MODEL ====
class LandmarkRegressor(nn.Module):
    def __init__(self, num_landmarks):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_landmarks * 3)

    def forward(self, x):
        return self.backbone(x)

# ==== LOAD ANNOTATION ====
with open(ANN_PATH, "r") as f:
    anno = json.load(f)

num_landmarks = len(anno["landmarks"])
gt_landmarks = np.array(anno["landmarks"], dtype=np.float32)  # already in pixels

# ==== LOAD MODEL ====
model = LandmarkRegressor(num_landmarks).to(DEVICE)
checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

# ==== TRANSFORM ====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ==== LOAD IMAGE ====
orig_pil = Image.open(IMG_PATH).convert("RGB")
orig_w, orig_h = orig_pil.size
inp = transform(orig_pil).unsqueeze(0).to(DEVICE)

# ==== INFERENCE ====
with torch.no_grad():
    preds = model(inp).cpu().numpy().reshape(-1, 3)

# ==== CONVERT NORMALIZED PREDICTIONS BACK TO PIXELS ====
# ==== CONVERT NORMALIZED PREDICTIONS BACK TO PIXELS (on 224x224 image) ====
pred_landmarks = preds.copy()
pred_landmarks[:, 0] *= IMG_SIZE
pred_landmarks[:, 1] *= IMG_SIZE


for idx, (px, py, pv) in enumerate(pred_landmarks):
    print(f"Landmark {idx}: Predicted (x={px:.1f}, y={py:.1f}, v={pv:.2f}) | ")

# ==== VISUALIZE ON RESIZED IMAGE ====
resized_pil = orig_pil.resize((IMG_SIZE, IMG_SIZE))
cv_img = cv2.cvtColor(np.array(resized_pil), cv2.COLOR_RGB2BGR)

for idx, ((px, py, pv), (gx, gy, gv)) in enumerate(zip(pred_landmarks, gt_landmarks)):
    # scale ground truth to 224x224 for comparison
    gx_scaled = gx / orig_w * IMG_SIZE
    gy_scaled = gy / orig_h * IMG_SIZE

    px, py = int(px), int(py)
    gx, gy = int(gx_scaled), int(gy_scaled)

    if gv == 0:
        continue

    # Ground truth = red
    #cv2.circle(cv_img, (gx, gy), 3, (0, 0, 255), -1)
    # Prediction = green
    cv2.circle(cv_img, (px, py), 3, (0, 255, 0), -1)
    
    cv2.putText(
        cv_img,
        str(idx + 1),               # landmark index
        (px, py - 5),           # slightly above the point
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,                    # font scale
        (255, 255, 255),        # white text
        1,                      # thickness
        cv2.LINE_AA
    )
    
    # Line between them = yellow
    #cv2.line(cv_img, (gx, gy), (px, py), (0, 255, 255), 1)

cv2.imshow("Predicted vs GT (224x224)", cv_img)
cv2.waitKey(0)
