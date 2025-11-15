import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# ==== CONFIG ====
PRETRAINED_PATH = "clothing_ai/checkpoints_backbone_resume/model_step_25000.pth"
NEW_ANNOS_DIR = "clothing_ai/data/annos"
NEW_IMAGES_DIR = "clothing_ai/data/images"

# --- Validation Data ---
VAL_ANNOS_DIR = "clothing_ai/data/val_annos"
VAL_IMAGES_DIR = "clothing_ai/data/val_images"

SAVE_DIR = "./checkpoints_backbone_v2_head"
LOG_DIR = "./runs/checkpoints_backbone_v2_head" # New log dir

# Improved Hyperparameters
EPOCHS = 150
UNFREEZE_EPOCH = 10
BATCH_SIZE = 16
NUM_NEW_LANDMARKS = 8
IMG_SIZE = 224
SAVE_EVERY_EPOCH = 30
DROPOUT_RATE = 0.5  # Dropout probability

# Differential Learning Rates
HEAD_LR = 1e-3
BACKBONE_LR = 1e-5

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

writer = SummaryWriter(LOG_DIR)

# ==== MASKED MSE LOSS (Updated) ====
def masked_mse_loss(preds, targets):
    # preds shape: [B, 8, 2]
    # targets shape: [B, 8, 3]
    
    # Get visibility mask from targets (the 3rd element)
    vis_mask = (targets[:, :, 2] > 0).unsqueeze(-1).float() # Shape: [B, 8, 1]
    
    # --- CHANGE 3: Compare preds [B, 8, 2] with targets [B, 8, 2] ---
    diff_sq = (preds - targets[:, :, :2]) ** 2 # Shape: [B, 8, 2]
    
    # Mask will broadcast to [B, 8, 2]
    loss = (diff_sq * vis_mask).sum() / vis_mask.sum().clamp(min=1)
    return loss

# ==== DATASET (No changes needed) ====
# Dataset MUST still return the (X, Y, V) tensor,
# so the loss function knows which points to ignore.
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
        print(f"✅ Loaded {len(self.samples)} samples.")

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

# ==== MODEL SETUP (Updated) ====
print(f"🔄 Loading backbone from {PRETRAINED_PATH}...")
base_model = models.resnet18(weights=None)
checkpoint = torch.load(PRETRAINED_PATH, map_location=device)
current_state = base_model.state_dict()
pretrained_backbone = {k: v for k, v in checkpoint.items() if k in current_state and 'fc' not in k}
current_state.update(pretrained_backbone)
base_model.load_state_dict(current_state)

for param in base_model.parameters():
    param.requires_grad = False

# --- Add Dropout + Linear Head ---
in_feats = base_model.fc.in_features
base_model.fc = nn.Sequential(
    nn.Dropout(p=DROPOUT_RATE),
    nn.Linear(in_feats, NUM_NEW_LANDMARKS * 2)
)
model = base_model.to(device)

# ==== TRAIN SETUP ====
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

train_dataset = NewSmallDataset(NEW_ANNOS_DIR, NEW_IMAGES_DIR, transform=train_tfms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Validation setup
val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

val_dataset = NewSmallDataset(VAL_ANNOS_DIR, VAL_IMAGES_DIR, transform=val_tfms)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Initial Optimizer
optimizer = optim.Adam(model.fc.parameters(), lr=HEAD_LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# ==== TRAINING LOOP (Updated) ====
best_train_loss = float('inf')
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    
    # --- UNFREEZE LOGIC ---
    if epoch == UNFREEZE_EPOCH:
        print(f"\n🔓 Epoch {epoch+1}: Unfreezing backbone with differential LR...")
        for param in model.parameters():
            param.requires_grad = True
            
        backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]
        head_params = model.fc.parameters()
        
        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': BACKBONE_LR},
            {'params': head_params, 'lr': HEAD_LR / 10}
        ])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # --- TRAINING ---
    model.train()
    running_train_loss = 0.0
    
    for imgs, landmarks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False):
        imgs = imgs.to(device)
        landmarks = landmarks.to(device)
        
        preds = model(imgs).view(-1, NUM_NEW_LANDMARKS, 2)
        loss = masked_mse_loss(preds, landmarks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    
    # --- VALIDATION ---
    model.eval()
    running_val_loss = 0.0
    
    with torch.no_grad():
        for imgs, landmarks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False):
            imgs = imgs.to(device)
            landmarks = landmarks.to(device)
            
            preds = model(imgs).view(-1, NUM_NEW_LANDMARKS, 2)
            loss = masked_mse_loss(preds, landmarks)
            
            running_val_loss += loss.item()
    
    avg_val_loss = running_val_loss / len(val_loader)
    
    # Update scheduler based on validation loss
    if epoch >= UNFREEZE_EPOCH:
        scheduler.step(avg_val_loss)

    # Logging
    writer.add_scalar("finetune_v2/train_mse_loss", avg_train_loss, epoch)
    writer.add_scalar("finetune_v2/val_mse_loss", avg_val_loss, epoch)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")

    # Save best model based on validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_finetuned_v2.pth"))
        print(f"🏆 New best model saved with val loss: {best_val_loss:.6f}")

    if (epoch + 1) % SAVE_EVERY_EPOCH == 0:
        ckpt_path = os.path.join(SAVE_DIR, f"finetuned_v2_ep{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")

torch.save(model.state_dict(), os.path.join(SAVE_DIR, "finetuned_v2_final.pth"))
writer.close()
print("🎉 Done!")
print(f"Best Validation Loss: {best_val_loss:.6f}")