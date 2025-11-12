import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import glob
import re
import matplotlib.pyplot as plt

# ==== CONFIG ====
# --- 1. SET VALIDATION DATA PATHS (Based on your structure) ---
VAL_ANNOS_DIR = "./data/val_annos"
VAL_IMAGES_DIR = "./data/val_images"

# --- 2. SET TRAINING DATA PATHS ---
TRAIN_ANNOS_DIR = "./data/annos"
TRAIN_IMAGES_DIR = "./data/images"

# --- 3. SET MODEL CHECKPOINT INFO ---
SAVE_DIR = "./checkpoints"
# This prefix MUST match your v1 training script
MODEL_PREFIX = "finetuned_v1" 
# Total epochs from your v1 script
EPOCHS_CONFIG = 500 

# --- 4. MODEL PARAMS (Must match head_train_v1.py) ---
NUM_NEW_LANDMARKS = 8
IMG_SIZE = 224
BATCH_SIZE = 16 # Use a larger batch size for faster evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==== MASKED L1 LOSS (Must match head_train_v1.py) ====
def masked_l1_loss(preds, targets):
    # preds shape: [B, 8, 2]
    # targets shape: [B, 8, 3]
    vis_mask = (targets[:, :, 2] > 0).unsqueeze(-1).float() # Shape: [B, 8, 1]
    diff = torch.abs(preds - targets[:, :, :2]) # Shape: [B, 8, 2]
    loss = (diff * vis_mask).sum() / vis_mask.sum().clamp(min=1)
    return loss

# ==== DATASET (Must match training) ====
class NewSmallDataset(Dataset):
    def __init__(self, annos_dir, images_dir, transform=None):
        self.transform = transform
        self.samples = []
        if not os.path.exists(annos_dir):
             raise FileNotFoundError(f"Annotations directory not found: {annos_dir}")
        for jf in sorted([f for f in os.listdir(annos_dir) if f.endswith('.json')]):
            with open(os.path.join(annos_dir, jf), 'r') as f:
                data = json.load(f)
            self.samples.append({
                'img_path': os.path.join(images_dir, data['image']),
                'landmarks': data['landmarks']
            })
        print(f"Loaded {len(self.samples)} samples from {annos_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
            item = self.samples[idx]
            try:
                img = Image.open(item['img_path']).convert("RGB")
            except FileNotFoundError:
                raise FileNotFoundError(f"Image missing: {item['img_path']}")
            landmarks = torch.tensor(item['landmarks'], dtype=torch.float32)
            landmarks[:, 0] /= 100.0
            landmarks[:, 1] /= 100.0
            if self.transform:
                img = self.transform(img)
            return img, landmarks

# ==== MODEL LOADER (Must match training) ====
def load_model_for_eval(model_path):
    model = models.resnet18(weights=None)
    # Output is XY-only (8 * 2 = 16)
    model.fc = nn.Linear(model.fc.in_features, NUM_NEW_LANDMARKS * 2)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None
    model.to(device)
    model.eval()
    return model

# ==== EVALUATION LOOP ====
def evaluate(model, loader, loss_fn):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for imgs, landmarks in tqdm(loader, desc="  Evaluating...", leave=False):
            imgs = imgs.to(device)
            landmarks = landmarks.to(device) # Shape: [B, 8, 3]
            
            preds = model(imgs).view(-1, NUM_NEW_LANDMARKS, 2) # Shape: [B, 8, 2]
            loss = loss_fn(preds, landmarks)
            running_loss += loss.item()
            
    return running_loss / len(loader)

# ==== MAIN SCRIPT ====
if __name__ == "__main__":
    
    # 1. Set up dataloaders
    # Use simple Resize/ToTensor (no augmentation) for validation
    val_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    print("Loading Training data for loss calculation...")
    train_dataset = NewSmallDataset(TRAIN_ANNOS_DIR, TRAIN_IMAGES_DIR, transform=val_tfms)
    print("Loading Validation data...")
    val_dataset = NewSmallDataset(VAL_ANNOS_DIR, VAL_IMAGES_DIR, transform=val_tfms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Set the correct loss function
    loss_fn = masked_l1_loss

    # 2. Find all checkpoints to test
    checkpoints_to_eval = []
    # Regex to find epoch number
    epoch_pattern = re.compile(rf"{MODEL_PREFIX}_ep(\d+)\.pth")
    
    print(f"\nScanning for models starting with '{MODEL_PREFIX}' in {SAVE_DIR}...")
    
    for f in glob.glob(os.path.join(SAVE_DIR, f"{MODEL_PREFIX}_ep*.pth")):
        match = epoch_pattern.search(f)
        if match:
            epoch = int(match.group(1))
            checkpoints_to_eval.append((epoch, f))
            
    # Add the final model
    final_model_path = os.path.join(SAVE_DIR, f"{MODEL_PREFIX}_final.pth")
    if os.path.exists(final_model_path):
        checkpoints_to_eval.append((EPOCHS_CONFIG, final_model_path))
        
    # Add the best model
    best_model_path = os.path.join(SAVE_DIR, f"best_{MODEL_PREFIX}.pth")
    
    # Sort by epoch
    checkpoints_to_eval.sort(key=lambda x: x[0])
    
    if not checkpoints_to_eval:
        print(f"Error: No models found with prefix '{MODEL_PREFIX}'. Check SAVE_DIR and MODEL_PREFIX.")
        exit()

    print(f"Found {len(checkpoints_to_eval)} epoch/final models to evaluate.")

    # 3. Run evaluation loop
    results = []
    for epoch, path in checkpoints_to_eval:
        print(f"\n--- Evaluating Epoch {epoch} ({os.path.basename(path)}) ---")
        model = load_model_for_eval(path)
        if model is None:
            continue
            
        train_loss = evaluate(model, train_loader, loss_fn)
        val_loss = evaluate(model, val_loader, loss_fn)
        
        print(f"  Train L1 Loss: {train_loss:.6f}")
        print(f"  Val L1 Loss:   {val_loss:.6f}")
        results.append({'epoch': epoch, 'train': train_loss, 'val': val_loss})

    # 4. Evaluate the 'best' model separately
    best_val_loss = float('inf')
    if os.path.exists(best_model_path):
        print(f"\n--- Evaluating BEST Model ({os.path.basename(best_model_path)}) ---")
        model = load_model_for_eval(best_model_path)
        best_val_loss = evaluate(model, val_loader, loss_fn)
        print(f"  Best Model Val L1 Loss: {best_val_loss:.6f}")

    # 5. Plot the results
    if results:
        epochs = [r['epoch'] for r in results]
        train_losses = [r['train'] for r in results]
        val_losses = [r['val'] for r in results]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
        plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
        
        # Plot the "best" model loss as a star
        if best_val_loss != float('inf'):
             # Find which epoch the best model came from (by finding min val loss)
             # Note: This assumes the 'best' model was saved based on training loss.
             # We plot its validation loss.
             plt.plot(epochs[-1], best_val_loss, 'g*', markersize=15, label=f'Best Model (Val Loss: {best_val_loss:.4f})')

        plt.title('Training vs. Validation Loss (v1 - L1)')
        plt.xlabel('Epoch')
        plt.ylabel('Masked L1 Loss')
        plt.legend()
        plt.grid(True)
        plt.xticks(epochs)
        plt.tight_layout()
        print("\nPlot window is open. Close it to exit.")
        plt.show()