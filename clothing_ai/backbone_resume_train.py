import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# ==== CONFIG ====
DATA_ROOT = "/home/peter/uni/clothing_ai/Data"
SAVE_DIR = "./checkpoints"
LOG_DIR = "./runs/landmark_logs"

BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 30             # number of *new* epochs to train
IMG_SIZE = 224
SAVE_EVERY = 1000
RESUME_CHECKPOINT = os.path.join(SAVE_DIR, "model_step_9000.pth")  # path to checkpoint

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

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

        landmarks = torch.tensor(anno["landmarks"], dtype=torch.float32)  # (num_landmarks, 3)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # normalize x,y coordinates
        landmarks[:, 0] /= w
        landmarks[:, 1] /= h

        if self.transform:
            img = self.transform(img)

        return img, landmarks


# ==== TRANSFORMS ====
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ==== DATALOADERS ====
train_dataset = ShortSleeveLandmarkDataset(os.path.join(DATA_ROOT, "train"), transform=train_tfms)
val_dataset = ShortSleeveLandmarkDataset(os.path.join(DATA_ROOT, "validation"), transform=val_tfms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ==== MODEL ====
class LandmarkRegressor(nn.Module):
    def __init__(self, num_landmarks):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_landmarks * 3)

    def forward(self, x):
        return self.backbone(x).view(-1, num_landmarks, 3)

# ==== INIT MODEL ====
first_ann_file = os.listdir(os.path.join(DATA_ROOT, "train", "annotations"))[0]
first_ann = json.load(open(os.path.join(DATA_ROOT, "train", "annotations", first_ann_file)))
num_landmarks = len(first_ann["landmarks"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LandmarkRegressor(num_landmarks).to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
writer = SummaryWriter(LOG_DIR)

# ==== MASKED L1 LOSS ====
def masked_l1_loss(preds, targets):
    vis_mask = (targets[:, :, 2] > 0).unsqueeze(-1).float()
    diff = torch.abs(preds[:, :, :2] - targets[:, :, :2])
    loss = (diff * vis_mask).sum() / vis_mask.sum().clamp(min=1)
    return loss

# ==== PIXEL ERROR METRIC ====
def mean_pixel_error(preds, targets, img_size):
    vis_mask = (targets[:, :, 2] > 0).unsqueeze(-1).float()
    diff = torch.abs(preds[:, :, :2] - targets[:, :, :2]) * vis_mask
    err = diff.sum(dim=-1) / 2.0
    mean_err = err.sum() / vis_mask.sum().clamp(min=1)
    return mean_err.item() * img_size

# ==== LOAD CHECKPOINT IF AVAILABLE ====
start_epoch = 0
global_step = 0
if os.path.exists(RESUME_CHECKPOINT):
    print(f"🔁 Resuming training from checkpoint: {RESUME_CHECKPOINT}")
    checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device)

    # Support for both "state_dict only" or full checkpoint
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("global_step", 0)
    else:
        # fallback if only state_dict was saved
        model.load_state_dict(checkpoint)
        print("⚠️ Checkpoint had no optimizer or epoch info — resuming model only.")
else:
    print("🚀 Starting training from scratch — no checkpoint found.")

# ==== TRAINING LOOP ====
for epoch in range(start_epoch, start_epoch + EPOCHS):
    model.train()
    running_train_loss = 0.0

    for imgs, landmarks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch+EPOCHS}"):
        imgs = imgs.to(device)
        landmarks = landmarks.to(device)

        preds = model(imgs)
        loss = masked_l1_loss(preds, landmarks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        writer.add_scalar("train/loss", loss.item(), global_step)
        global_step += 1

        if global_step % SAVE_EVERY == 0:
            ckpt_path = os.path.join(SAVE_DIR, f"model_step_{global_step}.pth")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

    avg_train_loss = running_train_loss / len(train_loader)
    print(f"✅ Epoch {epoch+1} | Train L1 Loss: {avg_train_loss:.4f}")

    # ---- VALIDATION ----
    model.eval()
    val_loss, val_pixel_err = 0.0, 0.0
    with torch.no_grad():
        for imgs, landmarks in val_loader:
            imgs, landmarks = imgs.to(device), landmarks.to(device)
            preds = model(imgs)
            val_loss += masked_l1_loss(preds, landmarks).item()
            val_pixel_err += mean_pixel_error(preds, landmarks, IMG_SIZE)

    avg_val_loss = val_loss / len(val_loader)
    avg_val_pixel_err = val_pixel_err / len(val_loader)
    writer.add_scalar("val/loss", avg_val_loss, epoch)
    writer.add_scalar("val/pixel_error", avg_val_pixel_err, epoch)
    print(f"📝 Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | Pixel Error: {avg_val_pixel_err:.2f}px")

# ==== FINAL SAVE ====
final_path = os.path.join(SAVE_DIR, "model_final_resumed.pth")
torch.save({
    "epoch": epoch,
    "global_step": global_step,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, final_path)

print(f"🎉 Training resumed and complete! Final model saved to {final_path}")
writer.close()
