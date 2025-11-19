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
import random
import glob

# ==== CONFIG ====
# Path to your fine-tuned 8-landmark model
MODEL_PATH = "clothing_ai/checkpoints_backbone_v2_head/best_finetuned_v2.pth" 
# ^^^ Make sure this matches your new model name (best_head_model_deep.pth)

NUM_LANDMARKS = 8
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== IMAGE FILES TO PROCESS ====
# Add image paths here
# If empty, random images will be selected from RANDOM_IMAGE_FOLDER
IMAGE_FILES = [
    # "clothing_ai/data/val_images/12_Color.png",
    # "clothing_ai/data/val_images/13_Color.png",
]

# ==== ANNOTATION FOLDER ====
# Folder where annotations are stored
# The program will automatically find annotations with matching filenames
# Set to None if you don't want to load any annotations
ANNOTATION_FOLDER = "clothing_ai/data/val_annos"

# ==== RANDOM IMAGE SELECTION (used when IMAGE_FILES is empty) ====
RANDOM_IMAGE_FOLDER = "clothing_ai/data/val_images"  # Folder to pick random images from
NUM_RANDOM_IMAGES = 6  # Number of random images to select

# ==== LOAD MODEL (Updated) ====
def load_model(model_path):
    print(f"⏳ Loading model from {model_path}...")
    base_model = models.resnet18(weights=None)
    in_feats = base_model.fc.in_features
    
    # --- Architecture for v2 model: Dropout + Linear ---
    # For v1 or deeper models, use the commented architecture below
    base_model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_feats, NUM_LANDMARKS * 2)
    )
    
    # --- Deeper head architecture (for other model versions) ---
    # Uncomment if using a model trained with deeper architecture
    # hidden_dim = 256
    # base_model.fc = nn.Sequential(
    #     nn.Linear(in_feats, hidden_dim),
    #     nn.ReLU(),
    #     nn.Dropout(p=0.5),
    #     nn.Linear(hidden_dim, NUM_LANDMARKS * 2)
    # )
    # --- End architecture options ---
    
    try:
        base_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit()
        
    model = base_model.to(DEVICE)
    model.eval() # Set model to evaluation mode (turns off dropout)
    print("✅ Model loaded successfully.")
    return model

# ==== HELPER: Get random images from folder ====
def get_random_images(img_folder, num_images):
    """Get random image paths from a folder"""
    if not os.path.exists(img_folder):
        print(f"⚠️ Folder not found: {img_folder}")
        return []
    
    # Get all image files (common formats)
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(img_folder, ext)))
    
    if not all_images:
        print(f"⚠️ No images found in {img_folder}")
        return []
    
    # Select random images
    num_to_select = min(num_images, len(all_images))
    selected_images = random.sample(all_images, num_to_select)
    
    print(f"🎲 Selected {num_to_select} random image(s) from {img_folder}")
    return selected_images

# ==== HELPER: Find matching annotation ====
def find_annotation(img_path, anno_folder):
    """Find annotation file that matches the image filename"""
    if not anno_folder or not os.path.exists(anno_folder):
        return None
    
    filename = os.path.basename(img_path)
    anno_filename = os.path.splitext(filename)[0] + '.json'
    anno_path = os.path.join(anno_folder, anno_filename)
    
    return anno_path if os.path.exists(anno_path) else None


# ==== INFERENCE FUNCTION ====
def predict_image(model, img_path, json_path=None):
    """Run inference on a single image and return results"""
    if not os.path.exists(img_path): 
        print(f"⚠️ Image not found: {img_path}")
        return None
        
    start_time = time.time()
    orig_img = Image.open(img_path).convert("RGB")
    w, h = orig_img.size
    print(f"📏 Image: {img_path} ({w}x{h})")

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

    print(f"⏱️  Inference time: {end_time-start_time:.4f}s")
    
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
                print(f"✅ Found annotations: {json_path}")
    
    return {
        'image': orig_img,
        'predictions': final_preds.tolist(),
        'ground_truth': gt_final,
        'path': img_path
    }

# ==== VISUALIZATION (Matplotlib) - Plot multiple images in a single window ====
def visualize_results(results):
    """Visualize inference results for multiple images in a single window"""
    num_images = len(results)
    if num_images == 0:
        print("⚠️ No results to visualize")
        return
    
    # Calculate grid layout (prefer wider layout)
    cols = min(3, num_images)  # Max 3 columns
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 8*rows))
    if num_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if num_images > 1 else [axes]
    
    for idx, result in enumerate(results):
        if result is None:
            continue
            
        ax = axes[idx]
        img = result['image']
        preds = result['predictions']
        gt = result['ground_truth']
        img_path = result['path']
        
        # Show image
        ax.imshow(img)
        ax.axis('off')
        
        # Extract filename for title
        filename = os.path.basename(img_path)
        ax.set_title(f"{filename}\n(Blue=Prediction | Green=GT)", fontsize=12)

        # Dynamic radius based on image size
        radius = max(5, int(min(img.size) * 0.008))

        # --- Draw Ground Truth (Green) ---
        if gt:
            print(f"\n--- Ground Truth for {filename} ---")
            for i, landmark_data in enumerate(gt):
                # Handle both 2D and 3D landmark formats
                if len(landmark_data) == 3:
                    px, py, pv = landmark_data
                else:
                    px, py = landmark_data
                    pv = 1.0  # Assume visible if no visibility value
                
                print(f"Landmark {i+1}: (x={px:.1f}, y={py:.1f}, v={pv:.2f})")
                
                if pv > 0:  # Only draw visible GT
                    # Draw Circle
                    circ = patches.Circle((px, py), radius=radius, linewidth=3, 
                                        edgecolor='green', facecolor='none')
                    ax.add_patch(circ)
                    
                    # Draw Text
                    ax.text(px - radius - 20, py, str(i+1), 
                            color='white', fontsize=12, weight='bold',
                            bbox=dict(facecolor='green', alpha=0.5, edgecolor='none', pad=1))

        # --- Draw Predictions (Blue) ---
        print(f"\n--- Predictions for {filename} ---")
        for i, (px, py) in enumerate(preds):
            print(f"Landmark {i+1}: (x={px:.1f}, y={py:.1f})")
            # Draw Circle
            circ = patches.Circle((px, py), radius=radius, linewidth=3, 
                                    edgecolor='blue', facecolor='none')
            ax.add_patch(circ)
            
            # Draw Text
            ax.text(px + radius + 5, py, str(i+1), 
                    color='white', fontsize=12, weight='bold',
                    bbox=dict(facecolor='blue', alpha=0.5, edgecolor='none', pad=1))
    
    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


# ==== RUN IT ====
if __name__ == '__main__':
    # 1. Initialize Model
    model = load_model(MODEL_PATH)

    # 2. Determine which images to process
    images_to_process = IMAGE_FILES
    
    if not images_to_process:  # If list is empty or None
        print(f"📁 No images specified, selecting random images...")
        images_to_process = get_random_images(RANDOM_IMAGE_FOLDER, NUM_RANDOM_IMAGES)
    
    if not images_to_process:
        print("❌ No images to process. Exiting.")
        exit()
    
    # 3. Process all images
    results = []
    print(f"\n🔄 Processing {len(images_to_process)} image(s)...\n")
    
    for img_path in images_to_process:
        # Automatically find matching annotation
        json_path = find_annotation(img_path, ANNOTATION_FOLDER)
        
        result = predict_image(model, img_path, json_path)
        if result:
            results.append(result)
        print()  # Add spacing between images
    
    # 4. Visualize all results in a single window
    if results:
        print(f"\n📊 Visualizing {len(results)} result(s)...\n")
        visualize_results(results)
    else:
        print("❌ No valid results to display")
