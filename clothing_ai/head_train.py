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
NEW_ANNOS_DIR = "./data/annos"
NEW_IMAGES_DIR = "./data/images"

SAVE_DIR = "./checkpoints"
LOG_DIR = "./runs/finetune_logs"

LR = 1e-5
EPOCHS = 300
BATCH_SIZE = 8
NUM_NEW_LANDMARKS = 8
IMG_SIZE = 224
SAVE_EVERY_EPOCH = 30

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

writer = SummaryWriter(LOG_DIR)

# ==== MASKED LOSS ====
def masked_l1_loss(preds, targets):
    vis_mask = (targets[:, :, 2] > 0).unsqueeze(-1).float()
    diff = torch.abs(preds[:, :, :2] - targets[:, :, :2])
    loss = (diff * vis_mask).sum() / vis_mask.sum().clamp(min=1)
    return loss

# ==== DATASET ====
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

            # No need for w, h if annotations are percentages
            landmarks = torch.tensor(item['landmarks'], dtype=torch.float32)

            # === CRITICAL FIX: NORMALIZE FROM PERCENTAGE ===
            # Input is 0-100, model needs 0.0-1.0
            landmarks[:, 0] /= 100.0
            landmarks[:, 1] /= 100.0
            # ===============================================

            if self.transform:
                img = self.transform(img)

            return img, landmarks

# ==== MODEL SETUP ====
print(f"🔄 Loading backbone from {PRETRAINED_PATH}...")
base_model = models.resnet18(weights=None)
checkpoint = torch.load(PRETRAINED_PATH, map_location=device)
current_state = base_model.state_dict()
pretrained_backbone = {k: v for k, v in checkpoint.items() if k in current_state and 'fc' not in k}
current_state.update(pretrained_backbone)
base_model.load_state_dict(current_state)

for param in base_model.parameters():
    param.requires_grad = False

base_model.fc = nn.Linear(base_model.fc.in_features, NUM_NEW_LANDMARKS * 3)
model = base_model.to(device)

# ==== TRAIN SETUP ====
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
])
dataset = NewSmallDataset(NEW_ANNOS_DIR, NEW_IMAGES_DIR, transform=train_tfms)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# ==== TRAINING LOOP ====
best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    if epoch == EPOCHS // 2:
        print("🔓 Unfreezing backbone...")
        for param in model.parameters():
            param.requires_grad = True
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR

    for imgs, landmarks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
        imgs, landmarks = imgs.to(device), landmarks.to(device)
        preds = model(imgs).view(-1, NUM_NEW_LANDMARKS, 3)
        loss = masked_l1_loss(preds, landmarks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    writer.add_scalar("finetune/train_loss", avg_loss, epoch)
    print(f"Epoch {epoch+1:02d} | Train Loss: {avg_loss:.4f}")

    # Save Best Model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_finetuned.pth"))

    # Periodic Save
    if (epoch + 1) % SAVE_EVERY_EPOCH == 0:
        ckpt_path = os.path.join(SAVE_DIR, f"finetuned_ep{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")

torch.save(model.state_dict(), os.path.join(SAVE_DIR, "finetuned_final.pth"))
writer.close()
print("🎉 Done!")