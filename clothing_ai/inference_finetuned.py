import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ==== CONFIG ====
# Path to your fine-tuned 8-landmark model
MODEL_PATH = "./checkpoints/best_finetuned.pth" 
# If best_finetuned.pth doesn't exist yet, use finetuned_final.pth

NUM_LANDMARKS = 8
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== LOAD MODEL ====
def load_model(model_path):
    model = models.resnet18(weights=None)
    # Ensure the head matches the fine-tuning script exactly
    model.fc = nn.Linear(model.fc.in_features, NUM_LANDMARKS * 3)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except FileNotFoundError:
        print(f"❌ Error: Model not found at {model_path}")
        exit()
        
    model.to(DEVICE)
    model.eval()
    return model

# ==== INFERENCE FUNCTION ====
def predict_image(model, img_path, json_path=None):
    if not os.path.exists(img_path): return

    orig_img = Image.open(img_path).convert("RGB")
    w, h = orig_img.size
    print(f"📏 Image: {w}x{h}")

    # --- PREDICT ---
    tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    img_tensor = tfms(orig_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # Model outputs [0.0 - 1.0]
        preds = model(img_tensor).view(NUM_LANDMARKS, 3).cpu()

    # Scale Model Output [0-1] -> Pixels
    final_preds = preds.clone()
    final_preds[:, 0] *= w
    final_preds[:, 1] *= h

    # --- LOAD GT (PERCENTAGE TO PIXEL) ---
    gt_final = None
    if json_path and os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            if 'landmarks' in data:
                gt = torch.tensor(data['landmarks'], dtype=torch.float32)
                # Convert GT from 0-100% to Pixels
                gt[:, 0] = (gt[:, 0] / 100.0) * w
                gt[:, 1] = (gt[:, 1] / 100.0) * h
                gt_final = gt.tolist()

    visualize_result(orig_img, final_preds.tolist(), gt_final)

# ==== VISUALIZATION (Matplotlib) ====
def visualize_result(img, preds, gt=None):
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Show image
    ax.imshow(img)
    ax.axis('off')
    plt.title("Green = GT | Red = Prediction", fontsize=16)

    # Dynamic radius based on image size
    radius = max(5, int(min(img.size) * 0.008))

    # --- Draw GT (Green) ---
    if gt:
        for i, pt in enumerate(gt):
             gx, gy = pt[0], pt[1]
             # Draw Circle using Matplotlib Patch
             circ = patches.Circle((gx, gy), radius=radius, linewidth=2, 
                                   edgecolor='#00FF00', facecolor='none')
             ax.add_patch(circ)

    # --- Draw Predictions (Red) ---
    print("\n--- Predictions ---")
    for i, (px, py, pv) in enumerate(preds):
        print(f"Landmark {i+1}: (x={px:.1f}, y={py:.1f}, v={pv:.2f})")
        
        if True:#pv > 0.1:
            # 1. Draw Circle
            circ = patches.Circle((px, py), radius=radius, linewidth=3, 
                                  edgecolor='red', facecolor='none')
            ax.add_patch(circ)
            
            # 2. Draw BIG Text next to it
            # fontsize=18 makes it much larger. Adjust as needed.
            ax.text(px + radius + 5, py + radius + 5, str(i+1), 
                    color='white', fontsize=12, weight='bold',
                    bbox=dict(facecolor='red', alpha=0.5, edgecolor='none', pad=1))

    plt.tight_layout()
    plt.show()

    # Optional: Save result
    # img.save("inference_result.png")

# ==== RUN IT ====
if __name__ == '__main__':
    # 1. Initialize Model
    model = load_model(MODEL_PATH)

    # 2. Define paths to test
    # OPTION A: Manual paths
    test_image = "./data/images/1_Color.png"
    test_json = "./data/annos/1_Color.json"

    # OPTION B: Auto-find from directory (uncomment to use)
    # image_dir = "./data/first/images"
    # annos_dir = "./data/first/annos"
    # first_img = os.listdir(image_dir)[0]
    # test_image = os.path.join(image_dir, first_img)
    # # Try to guess JSON path by replacing extension
    # likely_json = first_img.rsplit('.', 1)[0] + '.json'
    # test_json = os.path.join(annos_dir, likely_json)
    
    print(f"🖼️ Running inference on: {test_image}")
    if os.path.exists(test_json):
        print(f"📂 Found annotations: {test_json}")
    else:
        print("⚠️ No matching annotation file found (will only show predictions)")

    predict_image(model, test_image, test_json)