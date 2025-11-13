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
PRETRAINED_PATH = "./checkpoints/model_backbone.pth"
NEW_ANNOS_DIR = "./data/augmented_annos"
NEW_IMAGES_DIR = "./data/augmented_images"

SAVE_DIR = "./checkpoints"
# Updated log/save paths
LOG_DIR = "./runs/finetune_v1_reg_logs"
MODEL_SAVE_PREFIX = "finetuned_v1_reg"

LR = 1e-3
BACKBONE_LR = 1e-4
EPOCHS = 100
UNFREEZE_EPOCH = 100 # Your original script had this
BATCH_SIZE = 8
NUM_NEW_LANDMARKS = 8
IMG_SIZE = 224
SAVE_EVERY_EPOCH = 30
WEIGHT_DECAY = 1e-5 # Regularization parameter

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)
writer = SummaryWriter(LOG_DIR)

# ==== MASKED L1 LOSS (Unchanged) ====
def masked_l1_loss(preds, targets):
    vis_mask = (targets[:, :, 2] > 0).unsqueeze(-1).float() 
    diff = torch.abs(preds - targets[:, :, :2])
    loss = (diff * vis_mask).sum() / vis_mask.sum().clamp(min=1)
    return loss

# ==== DATASET (Unchanged) ====
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

# ==== MODEL SETUP (Updated with Dropout) ====
print(f"🔄 Loading backbone from {PRETRAINED_PATH}...")
base_model = models.resnet18(weights=None)
checkpoint = torch.load(PRETRAINED_PATH, map_location=device)
current_state = base_model.state_dict()
pretrained_backbone = {k: v for k, v in checkpoint.items() if k in current_state and 'fc' not in k}
current_state.update(pretrained_backbone)
base_model.load_state_dict(current_state)

for param in base_model.parameters():
    param.requires_grad = False

# --- CHANGE 1: Add Dropout layer ---
in_feats = base_model.fc.in_features
base_model.fc = nn.Sequential(
    nn.Dropout(p=0.5), # Add 50% dropout before the linear layer
    nn.Linear(in_feats, NUM_NEW_LANDMARKS * 2)
)
# --- END CHANGE ---

model = base_model.to(device)

# ==== TRAIN SETUP (Updated with Weight Decay) ====
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
])
dataset = NewSmallDataset(NEW_ANNOS_DIR, NEW_IMAGES_DIR, transform=train_tfms)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# --- CHANGE 2: Add weight_decay ---
optimizer = optim.Adam(model.fc.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ==== TRAINING LOOP (Updated with Weight Decay) ====
best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    if epoch == UNFREEZE_EPOCH:
        print("🔓 Unfreezing backbone...")
        for param in model.parameters():
            param.requires_grad = True
        
        backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]
        head_params = model.fc.parameters()
        
        # --- CHANGE 3: Add weight_decay ---
        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': BACKBONE_LR},
            {'params': head_params, 'lr': LR}
        ], weight_decay=WEIGHT_DECAY)

    for imgs, landmarks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
        imgs = imgs.to(device)
        landmarks = landmarks.to(device)
        
        preds = model(imgs).view(-1, NUM_NEW_LANDMARKS, 2)
        loss = masked_l1_loss(preds, landmarks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    writer.add_scalar(f"{MODEL_SAVE_PREFIX}/train_l1_loss", avg_loss, epoch)
    print(f"Epoch {epoch+1:02d} | Train L1 Loss: {avg_loss:.4f}")

    # Save Best Model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_{MODEL_SAVE_PREFIX}.pth"))
        print(f"🏆 New best model saved with loss: {best_loss:.4f}")

    # Periodic Save
    if (epoch + 1) % SAVE_EVERY_EPOCH == 0:
        ckpt_path = os.path.join(SAVE_DIR, f"{MODEL_SAVE_PREFIX}_ep{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")

torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{MODEL_SAVE_PREFIX}_final.pth"))
writer.close()
print("🎉 Done!")