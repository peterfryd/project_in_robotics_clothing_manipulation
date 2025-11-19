import os
import cv2
import torch
import numpy as np
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# ==== CONFIG ====
IMG_PATH = "/home/peter/uni/DeepFashion2/deepfashion2_original_images/validation/image/000647.jpg"
CKPT_PATH = "./checkpoints/model.pth"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_LANDMARKS = 25  # Set this to the number your model predicts

# ==== MODEL ====
class LandmarkRegressor(nn.Module):
    def __init__(self, num_landmarks):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_landmarks * 3)

    def forward(self, x):
        return self.backbone(x)

# ==== LOAD MODEL ====
model = LandmarkRegressor(NUM_LANDMARKS).to(DEVICE)
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
pred_landmarks = preds.copy()
pred_landmarks[:, 0] *= IMG_SIZE
pred_landmarks[:, 1] *= IMG_SIZE

# Print predictions
for idx, (px, py, pv) in enumerate(pred_landmarks):
    print(f"Landmark {idx}: Predicted (x={px:.1f}, y={py:.1f}, v={pv:.2f})")

# ==== VISUALIZE PREDICTIONS ====
resized_pil = orig_pil.resize((IMG_SIZE, IMG_SIZE))
cv_img = cv2.cvtColor(np.array(resized_pil), cv2.COLOR_RGB2BGR)

for idx, (px, py, pv) in enumerate(pred_landmarks):
    px, py = int(px), int(py)
    # Draw prediction points (green)
    cv2.circle(cv_img, (px, py), 3, (0, 255, 0), -1)
    # Put landmark index
    cv2.putText(
        cv_img,
        str(idx + 1),
        (px, py - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

cv2.imshow("Predicted Landmarks", cv_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
