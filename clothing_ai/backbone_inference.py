import os
import cv2
import json
import torch
import numpy as np
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# ==== CONFIG ====
IMAGE_NR = "000042"
IMG_PATH = "clothing_ai/data/Data_backbone/train/images/" + IMAGE_NR + ".jpg"
ANN_PATH = "clothing_ai/data/Data_backbone/train/annotations/" + IMAGE_NR + ".json"
CKPT_PATH = "clothing_ai/checkpoints_backbone_resume/model_backbone_original.pth"

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== MODEL ====
class LandmarkRegressor(nn.Module):
    def __init__(self, num_landmarks):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_landmarks * 3)

    def forward(self, x):
        return self.backbone(x).view(-1, num_landmarks, 3)

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

# ==== VISUALIZE ON LARGER IMAGE ====
# Scale up for better visualization (e.g., 3x larger = 672x672)
DISPLAY_SIZE = IMG_SIZE * 3
scale_factor = DISPLAY_SIZE / IMG_SIZE

resized_pil = orig_pil.resize((DISPLAY_SIZE, DISPLAY_SIZE))
cv_img = cv2.cvtColor(np.array(resized_pil), cv2.COLOR_RGB2BGR)

for idx, ((px, py, pv), (gx, gy, gv)) in enumerate(zip(pred_landmarks, gt_landmarks)):
    # Scale predictions and ground truth to display size
    px_display = int(px * scale_factor)
    py_display = int(py * scale_factor)
    
    gx_scaled = gx / orig_w * DISPLAY_SIZE
    gy_scaled = gy / orig_h * DISPLAY_SIZE
    gx_display = int(gx_scaled)
    gy_display = int(gy_scaled)

    if gv == 0:
        continue

    # Ground truth = red (uncomment to show)
    #cv2.circle(cv_img, (gx_display, gy_display), 5, (0, 0, 255), -1)
    # Prediction = green
    cv2.circle(cv_img, (px_display, py_display), 5, (0, 255, 0), -1)
    
    cv2.putText(
        cv_img,
        str(idx + 1),
        (px_display, py_display - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    
    # Line between them = yellow (uncomment to show)
    #cv2.line(cv_img, (gx_display, gy_display), (px_display, py_display), (0, 255, 255), 2)

# Create a named window with normal flags for better display
cv2.namedWindow("Landmark Predictions", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Landmark Predictions", DISPLAY_SIZE, DISPLAY_SIZE)
cv2.imshow("Landmark Predictions", cv_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
