import os
import cv2
import torch
import numpy as np
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import random
import glob

# ==== CONFIG ====
# Specify image path(s) here, or leave as None to use random images
# Can be a single path (string), a list of paths, or None
SINGLE_IMAGE_PATH = [
    # "clothing_ai/data/deepFashion2/validation/images/000072.jpg",
    # "clothing_ai/data/deepFashion2/validation/images/032098.jpg",
    # "clothing_ai/data/deepFashion2/validation/images/032097.jpg",
]
SINGLE_IMAGE_PATH = None
DATA_DIR = "clothing_ai/data/dino"  # Used if SINGLE_IMAGE_PATH is None
# DATA_DIR = "clothing_ai/data/deepFashion2/validation"
CKPT_PATH = "clothing_ai/checkpoints/model.pth"
NUM_IMAGES = 6  # Only used if SINGLE_IMAGE_PATH is None
NUM_LANDMARKS = 25  # Number of landmarks to predict

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_LANDMARKS = 25  # Set this to the number your model predicts

# ==== MODEL ====
class LandmarkRegressor(nn.Module):
    def __init__(self, num_landmarks):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_landmarks * 3)

    def forward(self, x):
        return self.backbone(x).view(-1, self.num_landmarks, 3)

# ==== GET RANDOM IMAGES ====
if SINGLE_IMAGE_PATH is not None:
    # Convert single path to list if needed
    if isinstance(SINGLE_IMAGE_PATH, str):
        selected_images = [SINGLE_IMAGE_PATH]
        print(f"Running inference on single image: {SINGLE_IMAGE_PATH}")
    else:
        selected_images = SINGLE_IMAGE_PATH
        print(f"Running inference on {len(selected_images)} specified images")
else:
    # Use random images - try multiple patterns and locations
    image_files = []
    
    # Try images subdirectory with jpg and png
    patterns = [
        os.path.join(DATA_DIR, "images", "*.jpg"),
        os.path.join(DATA_DIR, "images", "*.png"),
        os.path.join(DATA_DIR, "*.jpg"),
        os.path.join(DATA_DIR, "*.png"),
    ]
    
    for pattern in patterns:
        found = glob.glob(pattern)
        if found:
            image_files.extend(found)
            break  # Stop after finding images with first matching pattern
    
    if not image_files:
        raise FileNotFoundError(f"No images found in {DATA_DIR}. Tried patterns: {patterns}")
    
    selected_images = random.sample(image_files, min(NUM_IMAGES, len(image_files)))
    print(f"Running inference on {len(selected_images)} random images")

# ==== LOAD MODEL ====
print(f"Using {NUM_LANDMARKS} landmarks for model.")
model = LandmarkRegressor(NUM_LANDMARKS).to(DEVICE)
checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ==== TRANSFORM ====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ==== PROCESS ALL IMAGES ====
results = []

for img_path in selected_images:
    image_id = os.path.splitext(os.path.basename(img_path))[0]
    
    # Try to load annotation if it exists
    gt_landmarks = None
    if SINGLE_IMAGE_PATH is not None:
        img_dir = os.path.dirname(img_path)
        parent_dir = os.path.dirname(img_dir)
        ann_path = os.path.join(parent_dir, "annotations", f"{image_id}.json")
    else:
        ann_path = os.path.join(DATA_DIR, "annotations", f"{image_id}.json")
    
    if os.path.exists(ann_path):
        try:
            with open(ann_path, "r") as f:
                anno = json.load(f)
            gt_landmarks = np.array(anno["landmarks"], dtype=np.float32)
            print(f"Loaded annotations for {image_id}")
        except Exception as e:
            print(f"Warning: Could not load annotations for {image_id}: {e}")
    else:
        print(f"No annotations found for {image_id}, will only show predictions")
    
    # Load image
    orig_pil = Image.open(img_path).convert("RGB")
    orig_w, orig_h = orig_pil.size
    inp = transform(orig_pil).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        preds = model(inp).cpu().numpy().reshape(-1, 3)
    
    # Convert normalized predictions to pixels (on 224x224 image)
    pred_landmarks = preds.copy()
    pred_landmarks[:, 0] *= IMG_SIZE
    pred_landmarks[:, 1] *= IMG_SIZE
    
    results.append({
        'image_id': image_id,
        'orig_pil': orig_pil,
        'orig_w': orig_w,
        'orig_h': orig_h,
        'pred_landmarks': pred_landmarks,
        'gt_landmarks': gt_landmarks
    })

# ==== VISUALIZE ALL IMAGES IN A GRID ====
if len(results) == 1:
    # Single image - show larger
    DISPLAY_SIZE = IMG_SIZE * 3
    scale_factor = DISPLAY_SIZE / IMG_SIZE
    
    result = results[0]
    resized_pil = result['orig_pil'].resize((DISPLAY_SIZE, DISPLAY_SIZE))
    cv_img = cv2.cvtColor(np.array(resized_pil), cv2.COLOR_RGB2BGR)
    
    pred_landmarks = result['pred_landmarks']
    gt_landmarks = result['gt_landmarks']
    orig_w = result['orig_w']
    orig_h = result['orig_h']
    
    # Draw ground truth annotations first (green)
    if gt_landmarks is not None:
        for lm_idx, (gx, gy, gv) in enumerate(gt_landmarks):
            if gv == 0:
                continue
            
            # Scale ground truth from original image size to display size
            gx_scaled = gx / orig_w * DISPLAY_SIZE
            gy_scaled = gy / orig_h * DISPLAY_SIZE
            gx_display = int(gx_scaled)
            gy_display = int(gy_scaled)
            
            # Ground truth = green
            cv2.circle(cv_img, (gx_display, gy_display), 5, (0, 255, 0), -1)
            cv2.putText(
                cv_img,
                str(lm_idx + 1),
                (gx_display, gy_display - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
    
    # Draw predicted landmarks (blue)
    for lm_idx, (px, py, pv) in enumerate(pred_landmarks):
        px_display = int(px * scale_factor)
        py_display = int(py * scale_factor)
        
        # Prediction = blue
        cv2.circle(cv_img, (px_display, py_display), 5, (255, 0, 0), -1)
        cv2.putText(
            cv_img,
            str(lm_idx + 1),
            (px_display, py_display - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
    
    cv2.putText(
        cv_img,
        result['image_id'],
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )
    
    cv2.namedWindow("Landmark Predictions", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Landmark Predictions", DISPLAY_SIZE, DISPLAY_SIZE)
    cv2.imshow("Landmark Predictions", cv_img)
    
else:
    # Multiple images - show in grid
    DISPLAY_SIZE = IMG_SIZE * 2
    scale_factor = DISPLAY_SIZE / IMG_SIZE

    cols = 3
    rows = (len(results) + cols - 1) // cols

    grid_width = DISPLAY_SIZE * cols
    grid_height = DISPLAY_SIZE * rows
    grid_canvas = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    for idx, result in enumerate(results):
        resized_pil = result['orig_pil'].resize((DISPLAY_SIZE, DISPLAY_SIZE))
        cv_img = cv2.cvtColor(np.array(resized_pil), cv2.COLOR_RGB2BGR)
        
        pred_landmarks = result['pred_landmarks']
        gt_landmarks = result['gt_landmarks']
        orig_w = result['orig_w']
        orig_h = result['orig_h']
        
        # Draw ground truth annotations first (green)
        if gt_landmarks is not None:
            for lm_idx, (gx, gy, gv) in enumerate(gt_landmarks):
                if gv == 0:
                    continue
                
                # Scale ground truth from original image size to display size
                gx_scaled = gx / orig_w * DISPLAY_SIZE
                gy_scaled = gy / orig_h * DISPLAY_SIZE
                gx_display = int(gx_scaled)
                gy_display = int(gy_scaled)
                
                # Ground truth = green
                cv2.circle(cv_img, (gx_display, gy_display), 5, (0, 255, 0), -1)
                cv2.putText(
                    cv_img,
                    str(lm_idx + 1),
                    (gx_display, gy_display - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
        
        # Draw predicted landmarks (blue)
        for lm_idx, (px, py, pv) in enumerate(pred_landmarks):
            px_display = int(px * scale_factor)
            py_display = int(py * scale_factor)
            
            # Prediction = blue
            cv2.circle(cv_img, (px_display, py_display), 5, (255, 0, 0), -1)
            cv2.putText(
                cv_img,
                str(lm_idx + 1),
                (px_display, py_display - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        
        cv2.putText(
            cv_img,
            result['image_id'],
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        row = idx // cols
        col = idx % cols
        y_start = row * DISPLAY_SIZE
        y_end = y_start + DISPLAY_SIZE
        x_start = col * DISPLAY_SIZE
        x_end = x_start + DISPLAY_SIZE
        
        grid_canvas[y_start:y_end, x_start:x_end] = cv_img

    cv2.namedWindow("Landmark Predictions - All Images", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Landmark Predictions - All Images", grid_width, grid_height)
    cv2.imshow("Landmark Predictions - All Images", grid_canvas)

print(f"\nDisplaying {len(results)} image(s). Press any key to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()
