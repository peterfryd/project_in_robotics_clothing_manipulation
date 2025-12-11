import os
import cv2
import torch
import numpy as np
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import random
import glob
import json 
from tqdm import tqdm # <--- ADDED: Progress bar library

# ==== CONFIG ====
# Specify image path(s) here, or leave as None to use random images
SINGLE_IMAGE_PATH = [
    # "clothing_ai/data/deepFashion2/validation/images/000072.jpg",
    # "clothing_ai/data/deepFashion2/validation/images/032098.jpg",
    # "clothing_ai/data/deepFashion2/validation/images/032097.jpg",
]

# DATA_DIR should point to the parent directory containing 'images' and 'annotations' folders
DATA_DIR = "clothing_ai/data/deepFashion2/validation"

# Number of images to validate
NUM_IMAGES = 0

# Load model
CKPT_PATH = "clothing_ai/checkpoints/model.pth"


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

# ==== GET IMAGE PATHS (MODIFIED FOR FULL VALIDATION RUN) ====
is_full_run = False
if SINGLE_IMAGE_PATH is not None and len(SINGLE_IMAGE_PATH) > 0:
    # Specified images
    if isinstance(SINGLE_IMAGE_PATH, str):
        selected_images = [SINGLE_IMAGE_PATH]
    else:
        selected_images = SINGLE_IMAGE_PATH
    print(f"Running inference on {len(selected_images)} specified images")
    
else:
    # Full or Sample run
    image_files = []
    
    # Try finding images in the expected structure: DATA_DIR/images/*.jpg
    patterns = [
        os.path.join(DATA_DIR, "images", "*.jpg"),
        os.path.join(DATA_DIR, "images", "*.png"),
    ]
    
    for pattern in patterns:
        image_files.extend(glob.glob(pattern))
    
    if not image_files:
        raise FileNotFoundError(f"No images found in {DATA_DIR}. Tried patterns: {patterns}")
    
    if NUM_IMAGES is None or NUM_IMAGES <= 0:
        # Run on the entire dataset
        selected_images = image_files
        is_full_run = True
        print(f"Running inference on the entire dataset: {len(selected_images)} images.")
    else:
        # Sample run
        selected_images = random.sample(image_files, min(NUM_IMAGES, len(image_files)))
        print(f"Running inference on a sample of {len(selected_images)} random images.")

# If no images were found or selected, the script will exit gracefully later
if len(selected_images) == 0:
    print("Error: No images were selected to process. Check DATA_DIR and file paths.")

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

# ==== INITIALIZE AGGREGATION ARRAYS FOR AVERAGES ACROSS ALL IMAGES (NME) ====
total_landmark_NME = np.zeros(NUM_LANDMARKS)
landmark_counts = np.zeros(NUM_LANDMARKS, dtype=int)
all_NME_combined = []

# ==== PROCESS ALL IMAGES AND CALCULATE NME (NORMALIZED MEAN ERROR) ====
results = []
images_processed_with_gt = 0

# Use tqdm for progress bar if processing more than a handful of images
image_iterator = tqdm(selected_images, desc="Processing Images") if len(selected_images) > 5 else selected_images


for img_path in image_iterator:
    image_id = os.path.splitext(os.path.basename(img_path))[0]
    
    gt_landmarks = None
    
    # Determine the annotation path based on the DATA_DIR structure
    ann_path = os.path.join(DATA_DIR, "annotations", f"{image_id}.json")
    
    if os.path.exists(ann_path):
        try:
            with open(ann_path, "r") as f:
                anno = json.load(f)
            gt_landmarks = np.array(anno["landmarks"], dtype=np.float32)
            
            # Print image info only if it's NOT a full run
            if not is_full_run:
                 print(f"\n--- Processing {image_id} ---")
                 # Suppress this print for full run
                 # print(f"Loaded annotations for {image_id} from: {ann_path}")
        except Exception as e:
            if not is_full_run:
                print(f"Warning: Could not load annotations for {image_id}: {e}")
            gt_landmarks = None # Ensure GT is reset on failure
    
    # Skip processing if we couldn't load GT data for the purpose of evaluation
    if gt_landmarks is None:
        if not is_full_run:
            print(f"Skipping {image_id}: No valid annotations found.")
        continue

    images_processed_with_gt += 1
    
    # Load image
    orig_pil = Image.open(img_path).convert("RGB")
    orig_w, orig_h = orig_pil.size
    inp = transform(orig_pil).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        preds = model(inp).cpu().numpy().reshape(-1, 3)
    
    # Convert normalized predictions (relative to 224x224) to pixel coordinates for VISUALIZATION
    pred_landmarks_viz = preds.copy()
    pred_landmarks_viz[:, 0] *= IMG_SIZE
    pred_landmarks_viz[:, 1] *= IMG_SIZE
    
    # --- NME CALCULATION ---
    
    # 1. Determine Normalization Scale (Image Diagonal)
    scale = np.sqrt(orig_w**2 + orig_h**2)
    
    # 2. Scale predictions to the ORIGINAL image size
    pred_scaled_x = preds[:, 0] * orig_w
    pred_scaled_y = preds[:, 1] * orig_h
    
    # 3. Get Ground Truth coordinates and visibility
    gt_x = gt_landmarks[:, 0]
    gt_y = gt_landmarks[:, 1]
    gt_v = gt_landmarks[:, 2]

    # Calculate and aggregate NME
    for i in range(NUM_LANDMARKS):
        # Only calculate NME for visible or occluded landmarks (gv != 0)
        if gt_v[i] != 0:
            pixel_distance = np.sqrt((pred_scaled_x[i] - gt_x[i])**2 + (pred_scaled_y[i] - gt_y[i])**2)
            nme = pixel_distance / scale
            
            # --- AGGREGATION ---
            total_landmark_NME[i] += nme
            landmark_counts[i] += 1
            all_NME_combined.append(nme)
            
            if not is_full_run:
                # Suppress this print for full run
                print(f"Landmark {i+1:2d} NME: {nme:.4f} (Raw Pixels: {pixel_distance:.2f})")
    
    if not is_full_run:
        if len(all_NME_combined) > 0:
            # Print image average only if it's NOT a full run
            avg_nme_image = np.mean(all_NME_combined[-NUM_LANDMARKS:]) # approximate image avg
            print(f"\nImage Average NME: {avg_nme_image:.4f}")

    # Store results for visualization (only if it's a sample run)
    if not is_full_run:
        results.append({
            'image_id': image_id,
            'orig_pil': orig_pil,
            'orig_w': orig_w,
            'orig_h': orig_h,
            'pred_landmarks': pred_landmarks_viz,
            'gt_landmarks': gt_landmarks
        })


# ==== FINAL AGGREGATE RESULTS PRINT (NME SUMMARY) ====
print("\n" + "="*70)
print("             AGGREGATE NORMALIZED MEAN ERROR (NME) SUMMARY")
print(f"               Images Processed with GT: {images_processed_with_gt}")
print("               Normalization Scale: Image Diagonal")
print("="*70)

if np.sum(landmark_counts) > 0:
    print("\n--- Average NME per Landmark (Across All Images) ---")
    
    combined_valid_count = 0
    for i in range(NUM_LANDMARKS):
        count = landmark_counts[i]
        if count > 0:
            avg_nme = total_landmark_NME[i] / count
            print(f"Landmark {i+1:2d} Average NME: {avg_nme:.5f} (Count: {count})")
            combined_valid_count += count
        else:
            print(f"Landmark {i+1:2d} Average NME: N/A (Count: 0)")

    print("\n" + "-"*70)
    # Calculate and print the combined average across ALL valid NME measurements
    if all_NME_combined:
        overall_avg_nme = np.mean(all_NME_combined)
        print(f"COMBINED OVERALL AVERAGE NME: {overall_avg_nme:.5f} (Total Measurements: {combined_valid_count})")
    print("-" * 70)
else:
    print("No valid landmark measurements were collected to compute overall averages.")
    # Exit if no results
    exit()


# ==== VISUALIZATION BLOCK (Only runs if it was NOT a full run) ====
if is_full_run:
    print("\nVisualization skipped because a full dataset run was executed.")
    # Exit after printing aggregate results for full run
    exit() 

# --- If it's a sample run, proceed with visualization ---

if len(results) == 0:
    print("Exiting: No images were processed for visualization.")
# Continue only if results are available
elif len(results) == 1:
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
    
    # Draw ground truth annotations (GREEN)
    if gt_landmarks is not None:
        for lm_idx, (gx, gy, gv) in enumerate(gt_landmarks):
            if gv == 0: continue
            
            gx_scaled = gx / orig_w * DISPLAY_SIZE
            gy_scaled = gy / orig_h * DISPLAY_SIZE
            gx_display = int(gx_scaled)
            gy_display = int(gy_scaled)
            
            # Ground truth = Green (0, 255, 0)
            cv2.circle(cv_img, (gx_display, gy_display), 5, (0, 255, 0), -1)
            cv2.putText(cv_img, str(lm_idx + 1), (gx_display, gy_display - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Draw predicted landmarks (BLUE CROSS)
    for lm_idx, (px, py, pv) in enumerate(pred_landmarks):
        px_display = int(px * scale_factor)
        py_display = int(py * scale_factor)
        
        # Prediction = Blue (255, 0, 0)
        cv2.drawMarker(
            cv_img, 
            (px_display, py_display), 
            (255, 0, 0), # Blue color
            markerType=cv2.MARKER_CROSS,
            markerSize=10, 
            thickness=2
        )
        
        cv2.putText(cv_img, str(lm_idx + 1), (px_display, py_display - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.putText(cv_img, result['image_id'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    
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
        
        # Draw ground truth annotations (GREEN)
        if gt_landmarks is not None:
            for lm_idx, (gx, gy, gv) in enumerate(gt_landmarks):
                if gv == 0: continue
                
                gx_scaled = gx / orig_w * DISPLAY_SIZE
                gy_scaled = gy / orig_h * DISPLAY_SIZE
                gx_display = int(gx_scaled)
                gy_display = int(gy_scaled)
                
                # Ground truth = Green (0, 255, 0)
                cv2.circle(cv_img, (gx_display, gy_display), 5, (0, 255, 0), -1)
                cv2.putText(cv_img, str(lm_idx + 1), (gx_display, gy_display - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw predicted landmarks (BLUE CROSS)
        for lm_idx, (px, py, pv) in enumerate(pred_landmarks):
            px_display = int(px * scale_factor)
            py_display = int(py * scale_factor)
            
            # Prediction = Blue (255, 0, 0)
            cv2.drawMarker(
                cv_img, 
                (px_display, py_display), 
                (255, 0, 0), # Blue color
                markerType=cv2.MARKER_CROSS,
                markerSize=10, 
                thickness=2
            )
            
            cv2.putText(cv_img, str(lm_idx + 1), (px_display, py_display - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(cv_img, result['image_id'], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
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

if not is_full_run and len(results) > 0:
    print(f"\nDisplaying {len(results)} image(s). Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()