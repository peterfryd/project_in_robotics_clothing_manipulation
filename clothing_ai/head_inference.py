import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time

# ==== CONFIG ====
# Path to your fine-tuned 8-landmark model
MODEL_PATH = "./checkpoints/head_model_deep_best.pth" 
# ^^^ Make sure this matches your new model name (best_head_model_deep.pth)

NUM_LANDMARKS = 8
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== LOAD MODEL (Updated) ====
def load_model(model_path):
    print(f"⏳ Loading model from {model_path}...")
    base_model = models.resnet18(weights=None)
    in_feats = base_model.fc.in_features
    
    # --- Replicate the DEEPER head architecture from head_train.py ---
    hidden_dim = 256 # Must match the hidden_dim in your training script
    
    base_model.fc = nn.Sequential(
        nn.Linear(in_feats, hidden_dim), # 512 -> 256
        nn.ReLU(),                       # Activation
        nn.Dropout(p=0.5),               # Regularization
        nn.Linear(hidden_dim, NUM_NEW_LANDMARKS * 2) # 256 -> 16
    )
    # --- End architecture match ---
    
    try:
        base_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit()
        
    model = base_model.to(DEVICE)
    model.eval() # Set model to evaluation mode (turns off dropout)
    print("✅ Model loaded successfully.")
    return model

# ==== INFERENCE FUNCTION ====
def predict_image(model, img_path, json_path=None):
    if not os.path.exists(img_path): return
    start_time = time.time()
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
        preds = model(img_tensor).view(NUM_LANDMARKS, 2).cpu()
    end_time = time.time()

    print ("inference time: ",end_time-start_time)
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
                gt = torch.tensor(data['landmarks'], dtype=torch.float332)
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
        # Your updated GT drawing loop
        print("\n--- Ground Truth ---")
        for i, (px, py, pv) in enumerate(gt):
            print(f"Landmark {i+1}: (x={px:.1f}, y={py:.1f}, v={pv:.2f})")
            
            if pv > 0: # Only draw visible GT
                # 1. Draw Circle
                circ = patches.Circle((px, py), radius=radius, linewidth=3, 
                                    edgecolor='green', facecolor='none')
                ax.add_patch(circ)
                
                # 2. Draw Text
                ax.text(px - radius - 20 , py, str(i+1), 
                        color='white', fontsize=12, weight='bold',
                        bbox=dict(facecolor='green', alpha=0.5, edgecolor='none', pad=1))

    # --- Draw Predictions (Red) ---
    print("\n--- Predictions ---")
    for i, (px, py) in enumerate(preds):
        print(f"Landmark {i+1}: (x={px:.1f}, y={py:.1f}")
        # 1. Draw Circle
        circ = patches.Circle((px, py), radius=radius, linewidth=3, 
                                edgecolor='red', facecolor='none')
        ax.add_patch(circ)
        
        # 2. Draw Text
        ax.text(px + radius + 5, py, str(i+1), 
                color='white', fontsize=12, weight='bold',
                bbox=dict(facecolor='red', alpha=0.5, edgecolor='none', pad=1))

    plt.tight_layout()
    plt.show()

# ==== RUN IT ====
if __name__ == '__main__':
    # 1. Initialize Model
    model = load_model(MODEL_PATH)

    # 2. Define paths to test
    test_image = "./data/val_images/12_Color.png"
    test_json = "./data/val_annos/12_Color.json"
    
    print(f"🖼️ Running inference on: {test_image}")
    if os.path.exists(test_json):
        print(f"📂 Found annotations: {test_json}")
    else:
        print("⚠️ No matching annotation file found (will only show predictions)")

    predict_image(model, test_image, test_json)