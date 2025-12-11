import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==== CONFIG ====
DATA_ROOT = "clothing_ai/data/deepFashion2"
# Can be a single path (string) or a list of paths
CKPT_PATH = "clothing_ai/checkpoints_backbone_resume/model_step_25000.pth"
# CKPT_PATH = [
#     "clothing_ai/checkpoints_backbone_resume/model_backbone_original.pth",
#     "clothing_ai/checkpoints_backbone_resume/model_step_16000.pth",
#     "clothing_ai/checkpoints_backbone_resume/model_step_17000.pth",
#     "clothing_ai/checkpoints_backbone_resume/model_step_18000.pth",
#     "clothing_ai/checkpoints_backbone_resume/model_step_19000.pth",
#     "clothing_ai/checkpoints_backbone_resume/model_step_20000.pth",
#     "clothing_ai/checkpoints_backbone_resume/model_step_21000.pth",
#     "clothing_ai/checkpoints_backbone_resume/model_step_22000.pth"
#     "clothing_ai/checkpoints_backbone_resume/model_step_23000.pth"
#     "clothing_ai/checkpoints_backbone_resume/model_step_24000.pth"
#     "clothing_ai/checkpoints_backbone_resume/model_step_25000.pth"
# ]
IMG_SIZE = 224
BATCH_SIZE = 16 

# ==== DATASET ====
class ShortSleeveLandmarkDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_dir = os.path.join(root_dir, "images")
        self.ann_dir = os.path.join(root_dir, "annotations")
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith((".png", ".jpg"))])
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(self.ann_dir, img_name.replace(".png", ".json").replace(".jpg", ".json"))

        with open(ann_path, "r") as f:
            anno = json.load(f)

        landmarks = torch.tensor(anno["landmarks"], dtype=torch.float32)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        landmarks[:, 0] /= w
        landmarks[:, 1] /= h

        if self.transform:
            img = self.transform(img)

        landmarks = landmarks.flatten()
        return img, landmarks


# ==== MODEL ====
class LandmarkRegressor(nn.Module):
    def __init__(self, num_landmarks):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_landmarks * 3)

    def forward(self, x):
        return self.backbone(x)


# ==== LOAD DATA ====
val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

val_dataset = ShortSleeveLandmarkDataset(os.path.join(DATA_ROOT, "validation"), transform=val_tfms)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ==== INIT MODEL ====
first_ann = json.load(open(os.path.join(DATA_ROOT, "validation", "annotations",
                        os.listdir(os.path.join(DATA_ROOT, "validation", "annotations"))[0])))
num_landmarks = len(first_ann["landmarks"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== CONVERT CKPT_PATH TO LIST ====
if isinstance(CKPT_PATH, str):
    checkpoint_paths = [CKPT_PATH]
else:
    checkpoint_paths = CKPT_PATH

# ==== VALIDATION FUNCTION ====
def validate_model(ckpt_path):
    """Validate a single model checkpoint and return the loss."""
    model = LandmarkRegressor(num_landmarks=num_landmarks)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    criterion = nn.MSELoss()
    val_loss = 0.0
    
    total_images = len(val_loader.dataset)
    processed = 0

    with torch.no_grad():
        pbar = tqdm(total=total_images, desc=f"Validating {os.path.basename(ckpt_path)}", unit='img')
        for imgs, landmarks in val_loader:
            imgs, landmarks = imgs.to(device), landmarks.to(device)
            preds = model(imgs)
            # accumulate batch loss (keeping the same averaging behaviour as before)
            val_loss += criterion(preds, landmarks).item()

            # update progress counters and progress bar (show images processed / remaining)
            batch_count = imgs.size(0)
            processed += batch_count
            remaining = max(0, total_images - processed)
            pbar.update(batch_count)
            pbar.set_postfix({'processed': f"{processed}/{total_images}", 'left': remaining})

        pbar.close()

    # average over number of batches (same as original behaviour)
    val_loss /= len(val_loader)
    return val_loss

# ==== RUN VALIDATION ON ALL CHECKPOINTS ====
results = []

for ckpt_path in checkpoint_paths:
    print(f"\n{'='*60}")
    print(f"Validating: {ckpt_path}")
    print(f"{'='*60}")
    
    val_loss = validate_model(ckpt_path)
    
    results.append({
        'path': ckpt_path,
        'name': os.path.basename(ckpt_path),
        'loss': val_loss
    })
    
    print(f"✅ Validation MSE Loss: {val_loss:.6f}")

# ==== PRINT SUMMARY ====
print(f"\n{'='*60}")
print("VALIDATION SUMMARY")
print(f"{'='*60}")
for result in results:
    print(f"{result['name']:40s} | Loss: {result['loss']:.6f}")

# ==== PLOT COMPARISON IF MULTIPLE MODELS ====
if len(results) > 1:
    print(f"\n{'='*60}")
    print("Creating comparison plot...")
    print(f"{'='*60}")
    
    names = [r['name'] for r in results]
    losses = [r['loss'] for r in results]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(names)), losses, color='steelblue', alpha=0.7)
    plt.xlabel('Model Checkpoint', fontsize=12)
    plt.ylabel('Validation MSE Loss', fontsize=12)
    plt.title('Model Comparison - Validation Loss', fontsize=14, fontweight='bold')
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = "clothing_ai/validation_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {plot_path}")
    
    # Show plot
    plt.show()
    
    # Find best model
    best_result = min(results, key=lambda x: x['loss'])
    print(f"\n🏆 Best Model: {best_result['name']} with Loss: {best_result['loss']:.6f}")
else:
    print(f"\n✅ Single model validation complete.")
