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
from tqdm import tqdm

# ==== CONFIGURATION ==========================================================
# Specify image path(s) here, or leave as None to use random images
SINGLE_IMAGE_PATH = [
    # "clothing_ai/data/deepFashion2/validation/images/000072.jpg",
]

# DATA_DIR should point to the parent directory containing 'images' and 'annotations' folders
DATA_DIR = "clothing_ai/data/deepFashion2/validation"

# Load model
CKPT_PATH = "clothing_ai/checkpoints/model.pth"

# Number of best and worst images to plot for diagnosis (N)
N_SAMPLES_TO_PLOT = 10

# Output directory where the visualized best/worst images (4N total) will be saved
OUTPUT_DIR = "diagnostic_output" 

# Number of image to val on. If not a positive number then use all
NUM_IMAGES = 0

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_LANDMARKS = 25

# Landmarks subset for specialized combined average (0-indexed)
LANDMARK_SUBSET_INDICES = [16, 14, 9, 8, 1, 5, 22, 23]

# ==== MODEL DEFINITION (Unchanged) ===========================================
class LandmarkRegressor(nn.Module):
    def __init__(self, num_landmarks):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_landmarks * 3)

    def forward(self, x):
        return self.backbone(x).view(-1, self.num_landmarks, 3)

# ==== INITIALIZATION (Unchanged) =============================================
is_full_run = False
if SINGLE_IMAGE_PATH and len(SINGLE_IMAGE_PATH) > 0:
    selected_images = SINGLE_IMAGE_PATH if isinstance(SINGLE_IMAGE_PATH, list) else [SINGLE_IMAGE_PATH]
else:
    image_files = []
    for pattern in [os.path.join(DATA_DIR, "images", "*.jpg"), os.path.join(DATA_DIR, "images", "*.png")]:
        image_files.extend(glob.glob(pattern))
    if not image_files: raise FileNotFoundError(f"No images found in {DATA_DIR}.")
    if NUM_IMAGES is None or NUM_IMAGES <= 0:
        selected_images = image_files
        is_full_run = True
    else:
        selected_images = random.sample(image_files, min(NUM_IMAGES, len(image_files)))

model = LandmarkRegressor(NUM_LANDMARKS).to(DEVICE)
checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

all_landmark_NME_values = [[] for _ in range(NUM_LANDMARKS)]
all_NME_combined = []
subset_nme_combined = [] 
all_landmark_224_pix_values = [[] for _ in range(NUM_LANDMARKS)]
all_224_pix_combined = []
subset_224_pix_combined = [] 
image_nme_results = [] 

# ==== MAIN PROCESSING LOOP (Unchanged) =======================================
images_processed_with_gt = 0
image_iterator = tqdm(selected_images, desc="Processing Images") if len(selected_images) > 5 else selected_images

for img_path in image_iterator:
    image_id = os.path.splitext(os.path.basename(img_path))[0]
    ann_path = os.path.join(DATA_DIR, "annotations", f"{image_id}.json")
    
    gt_landmarks = None
    if os.path.exists(ann_path):
        try:
            with open(ann_path, "r") as f:
                anno = json.load(f)
            gt_landmarks = np.array(anno["landmarks"], dtype=np.float32)
        except Exception:
            gt_landmarks = None
    
    if gt_landmarks is None: continue

    images_processed_with_gt += 1
    
    orig_pil = Image.open(img_path).convert("RGB")
    orig_w, orig_h = orig_pil.size
    
    inp = transform(orig_pil).unsqueeze(0).to(DEVICE)
    orig_pil.close() 
    
    with torch.no_grad():
        preds = model(inp).cpu().numpy().reshape(-1, 3)
    
    gt_v = gt_landmarks[:, 2]
    scale_nme = np.sqrt(orig_w**2 + orig_h**2)
    pred_scaled_x_orig = preds[:, 0] * orig_w
    pred_scaled_y_orig = preds[:, 1] * orig_h
    nme_values_for_image = []

    gt_scaled_x_224 = gt_landmarks[:, 0] * (IMG_SIZE / orig_w)
    gt_scaled_y_224 = gt_landmarks[:, 1] * (IMG_SIZE / orig_h)
    pred_scaled_x_224 = preds[:, 0] * IMG_SIZE 
    pred_scaled_y_224 = preds[:, 1] * IMG_SIZE 

    for i in range(NUM_LANDMARKS):
        if gt_v[i] == 2: 
            pixel_distance_orig = np.sqrt((pred_scaled_x_orig[i] - gt_landmarks[i, 0])**2 + (pred_scaled_y_orig[i] - gt_landmarks[i, 1])**2)
            nme = pixel_distance_orig / scale_nme
            
            all_landmark_NME_values[i].append(nme)
            all_NME_combined.append(nme)
            if i in LANDMARK_SUBSET_INDICES: subset_nme_combined.append(nme)
            nme_values_for_image.append(nme)
            
            pix_distance_224 = np.sqrt(
                (pred_scaled_x_224[i] - gt_scaled_x_224[i])**2 + 
                (pred_scaled_y_224[i] - gt_scaled_y_224[i])**2
            )

            all_landmark_224_pix_values[i].append(pix_distance_224)
            all_224_pix_combined.append(pix_distance_224)
            if i in LANDMARK_SUBSET_INDICES: subset_224_pix_combined.append(pix_distance_224)

    current_image_avg_nme = np.mean(nme_values_for_image) if nme_values_for_image else 0
    
    image_nme_results.append({
        'image_id': image_id,
        'avg_nme': current_image_avg_nme,
        'img_path': img_path,
        'ann_path': ann_path,
        'orig_w': orig_w,
        'orig_h': orig_h,
        'preds': preds, 
    })


# ==== FINAL AGGREGATE RESULTS PRINT BLOCK (RE-ADDED) ==============================

def print_aggregate_stats(metric_name, values_list, is_nme=False):
    """Helper function for printing aggregated stats."""
    if len(values_list) > 0:
        avg = np.mean(values_list)
        var = np.var(values_list)
        
        format_str = ".5f" if is_nme else ".2f"
        unit_str = "" if is_nme else " px"
        
        print(f"COMBINED OVERALL AVERAGE {metric_name}: {avg:{format_str}}{unit_str}")
        print(f"COMBINED OVERALL VARIANCE {metric_name}: {var:{format_str}}{unit_str} (Total Measurements: {len(values_list)})")
    else:
        print(f"No measurements collected for {metric_name}.")
    print("-" * 70)


print("\n" + "="*80)
print("             AGGREGATE PERFORMANCE SUMMARY (ANALYSIS ONLY INCLUDES v=2 LANDMARKS)")
print(f"                       Images Processed with GT: {images_processed_with_gt}")
print("="*80)

# --- 1. NME ANALYSIS (FULL SET) ---
print("\n--- NME Analysis (Normalization Scale: Image Diagonal) ---")
print("\n--- Mean and Variance per Landmark (NME) ---")

if len(all_NME_combined) > 0:
    for i in range(NUM_LANDMARKS):
        nme_list = all_landmark_NME_values[i]
        count = len(nme_list)
        if count > 0:
            nme_mean = np.mean(nme_list)
            nme_variance = np.var(nme_list)
            print(f"Landmark {i+1:2d}: Mean={nme_mean:.5f}, Variance={nme_variance:.5f} (Count: {count})")
        else:
            print(f"Landmark {i+1:2d}: N/A (Count: 0)")

    print("\n" + "-"*70)
    print_aggregate_stats("NME (FULL SET)", all_NME_combined, is_nme=True)
else:
    print("No valid NME measurements were collected.")

# --- 2. 224x224 PIXEL DISTANCE ANALYSIS (FULL SET) ---
print("\n--- 224x224 Pixel Distance Analysis ---")
print("\n--- Mean and Variance per Landmark (224 Pixels) ---")

if len(all_224_pix_combined) > 0:
    for i in range(NUM_LANDMARKS):
        pix_list = all_landmark_224_pix_values[i]
        count = len(pix_list)
        if count > 0:
            pix_mean = np.mean(pix_list)
            pix_variance = np.var(pix_list)
            print(f"Landmark {i+1:2d}: Mean={pix_mean:.2f} px, Variance={pix_variance:.2f} (Count: {count})")
        else:
            print(f"Landmark {i+1:2d}: N/A (Count: 0)")

    print("\n" + "-"*70)
    print_aggregate_stats("224 PIXELS (FULL SET)", all_224_pix_combined, is_nme=False)
else:
    print("No valid 224x224 pixel distance measurements were collected.")


# --- 3. SUBSET ANALYSIS ---
subset_1_indexed = [i+1 for i in LANDMARK_SUBSET_INDICES]
print("\n" + "="*80)
print(f"    SUBSET ANALYSIS: Landmarks {subset_1_indexed} (Analysis Only Includes v=2)")
print("="*80)

# Individual Landmark NME Stats
print("\n--- Subset NME (Individual Landmark Mean/Variance) ---")
for i in LANDMARK_SUBSET_INDICES:
    nme_list = all_landmark_NME_values[i]
    count = len(nme_list)
    
    if count > 0:
        nme_mean = np.mean(nme_list)
        nme_variance = np.var(nme_list)
        print(f"Landmark {i+1:2d}: Mean={nme_mean:.5f}, Variance={nme_variance:.5f} (Count: {count})")
    else:
        print(f"Landmark {i+1:2d}: N/A (Count: 0)")

print("\n--- Subset NME (Combined) ---")
print_aggregate_stats("NME (SUBSET)", subset_nme_combined, is_nme=True)


# Individual Landmark 224 Pixel Stats
print("\n--- Subset 224x224 Pixels (Individual Landmark Mean/Variance) ---")
for i in LANDMARK_SUBSET_INDICES:
    pix_list = all_landmark_224_pix_values[i]
    count = len(pix_list)
    
    if count > 0:
        pix_mean = np.mean(pix_list)
        pix_variance = np.var(pix_list)
        print(f"Landmark {i+1:2d}: Mean={pix_mean:.2f} px, Variance={pix_variance:.2f} (Count: {count})")
    else:
        print(f"Landmark {i+1:2d}: N/A (Count: 0)")

print("\n--- Subset 224x224 Pixels (Combined) ---")
print_aggregate_stats("224 PIXELS (SUBSET)", subset_224_pix_combined, is_nme=False)


# ==== DIAGNOSTIC VISUALIZATION AND SAVING BLOCK (Unchanged) =====================

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if images_processed_with_gt >= 4 * N_SAMPLES_TO_PLOT:
    
    N_TO_PRINT = 2 * N_SAMPLES_TO_PLOT 
    plottable_results = [r for r in image_nme_results if r['avg_nme'] > 0]
    plottable_results.sort(key=lambda x: x['avg_nme'])
    best_samples_full = plottable_results[:N_TO_PRINT]
    worst_samples_full = plottable_results[-N_TO_PRINT:][::-1]
    best_samples_display = best_samples_full[:N_SAMPLES_TO_PLOT]
    worst_samples_display = worst_samples_full[:N_SAMPLES_TO_PLOT]
    combined_samples_display = best_samples_display + worst_samples_display
    display_image_ids = {r['image_id'] for r in combined_samples_display}
    combined_samples_save = best_samples_full + worst_samples_full

    # --- Print block for file names is here (omitted for brevity) ---
    print(f"\n--- Top {N_TO_PRINT} Best Images (Lowest NME) ---")
    for i, result in enumerate(best_samples_full):
        print(f"RANK {i+1:<3}: {result['image_id']}.jpg (Avg NME: {result['avg_nme']:.5f})")
    print(f"\n--- Top {N_TO_PRINT} Worst Images (Highest NME) ---")
    for i, result in enumerate(worst_samples_full):
        print(f"RANK {i+1:<3}: {result['image_id']}.jpg (Avg NME: {result['avg_nme']:.5f}")
    print(f"\n--- Printed {4 * N_SAMPLES_TO_PLOT} file names in total ---")
    print("-" * 80)
    # -----------------------------------------------------------------

    DISPLAY_SIZE = IMG_SIZE * 2
    cols = min(N_SAMPLES_TO_PLOT, len(combined_samples_display))
    rows = 2
    grid_width = DISPLAY_SIZE * cols
    grid_height = DISPLAY_SIZE * rows
    grid_canvas = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    print(f"\nSaving {len(combined_samples_save)} visualized images to '{OUTPUT_DIR}'...")
    
    processed_images_for_grid = {} 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    font_scale = 0.4
    font_thickness = 1
    outline_thickness = 3

    for idx, result in enumerate(tqdm(combined_samples_save, desc="Loading, Visualizing, and Saving Images")):
        
        with open(result["ann_path"], "r") as f:
            anno = json.load(f)
        gt_landmarks = np.array(anno["landmarks"], dtype=np.float32)

        orig_pil = Image.open(result['img_path']).convert("RGB")
        resized_pil = orig_pil.resize((DISPLAY_SIZE, DISPLAY_SIZE))
        orig_pil.close() 
        cv_img = cv2.cvtColor(np.array(resized_pil), cv2.COLOR_RGB2BGR)
        resized_pil.close() 
        
        pred_landmarks = result['preds'].copy()
        pred_landmarks[:, 0] *= DISPLAY_SIZE
        pred_landmarks[:, 1] *= DISPLAY_SIZE

        orig_w = result['orig_w']
        orig_h = result['orig_h']
        
        is_best = idx < N_TO_PRINT 
        prefix = "best_"
        if not is_best: prefix = "worst_"
        
        # --- Draw Ground Truth (Green Circle) ---
        if gt_landmarks is not None:
            for lm_idx, (gx, gy, gv) in enumerate(gt_landmarks):
                
                if gv != 2: continue 
                
                gx_scaled = gx / orig_w * DISPLAY_SIZE
                gy_scaled = gy / orig_h * DISPLAY_SIZE
                gt_x_disp = int(gx_scaled)
                gt_y_disp = int(gy_scaled)

                cv2.circle(cv_img, (gt_x_disp, gt_y_disp), 5, (0, 255, 0), -1)
                
                # ADDED: Landmark number with Black outline and White text
                text_pos = (gt_x_disp, gt_y_disp - 8)
                
                # Outline (Black, Thicker)
                cv2.putText(cv_img, str(lm_idx + 1), text_pos, font, font_scale, (0, 0, 0), outline_thickness, cv2.LINE_AA)
                # Text (White, Thinner)
                cv2.putText(cv_img, str(lm_idx + 1), text_pos, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # --- Draw Predicted Landmarks (Blue Cross / Yellow) ---
        if gt_landmarks is not None:
            for lm_idx, (px, py, pv) in enumerate(pred_landmarks):
                
                gv = gt_landmarks[lm_idx, 2] 
                
                if gv == 2:
                    marker_color = (255, 0, 0) # Blue (BGR)
                else:
                    marker_color = (0, 255, 255) # Yellow (BGR)
                    
                px_display = int(px)
                py_display = int(py)

                cv2.drawMarker(
                    cv_img, 
                    (px_display, py_display), 
                    marker_color,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=10, 
                    thickness=2
                )
                
                # ADDED: Landmark number with Black outline and White text
                text_pos = (px_display, py_display - 8)

                # Outline (Black, Thicker)
                cv2.putText(cv_img, str(lm_idx + 1), text_pos, font, font_scale, (0, 0, 0), outline_thickness, cv2.LINE_AA)
                # Text (White, Thinner)
                cv2.putText(cv_img, str(lm_idx + 1), text_pos, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # --- SAVING THE IMAGE ---
        save_filename = f"{prefix}{result['image_id']}.jpg"
        save_path = os.path.join(OUTPUT_DIR, save_filename)
        cv2.imwrite(save_path, cv_img)
        
        if result['image_id'] in display_image_ids:
            processed_images_for_grid[result['image_id']] = cv_img

    print(f"✅ Successfully saved {len(combined_samples_save)} visualized images with 'best_' / 'worst_' prefixes to '{OUTPUT_DIR}'.")
    
    # --- LOOP 2: CREATE AND DISPLAY THE GRID (Unchanged) ---
    print(f"\nDisplaying Top {N_SAMPLES_TO_PLOT} BEST and Top {N_SAMPLES_TO_PLOT} WORST samples in a grid.")
    
    for idx, result in enumerate(combined_samples_display):
        cv_img_processed = processed_images_for_grid.get(result['image_id'])
        if cv_img_processed is None:
            continue 

        row = 0 if idx < N_SAMPLES_TO_PLOT else 1
        col = idx % N_SAMPLES_TO_PLOT
        y_start = row * DISPLAY_SIZE
        y_end = y_start + DISPLAY_SIZE
        x_start = col * DISPLAY_SIZE
        x_end = x_start + DISPLAY_SIZE
        
        grid_canvas[y_start:y_end, x_start:x_end] = cv_img_processed

    cv2.namedWindow(f"Diagnostic Visualization ({N_SAMPLES_TO_PLOT} Best & Worst)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f"Diagnostic Visualization ({N_SAMPLES_TO_PLOT} Best & Worst)", grid_width, grid_height)
    cv2.imshow(f"Diagnostic Visualization ({N_SAMPLES_TO_PLOT} Best & Worst)", grid_canvas)

    print(f"\nDisplaying {2 * N_SAMPLES_TO_PLOT} diagnostic image(s) grid. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()