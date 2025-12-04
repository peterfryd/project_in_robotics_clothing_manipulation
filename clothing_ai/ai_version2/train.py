# train_df2_cached.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.tensorboard import SummaryWriter

CACHE_DIR = "./df2_cache"
BATCH_SIZE = 4
NUM_EPOCHS = 20
NUM_KEYPOINTS = 25
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ================================
# CACHED DATASET
# ================================
class CachedDataset(Dataset):
    def __init__(self, cache_folder):
        self.files = sorted([os.path.join(cache_folder, f) for f in os.listdir(cache_folder) if f.endswith(".pt")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx])

train_dataset = CachedDataset(os.path.join(CACHE_DIR, "train"))
val_dataset = CachedDataset(os.path.join(CACHE_DIR, "val"))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# ================================
# MODEL
# ================================
model = keypointrcnn_resnet50_fpn(weights=None, num_keypoints=NUM_KEYPOINTS)

# Box head (2 classes: background + short sleeve top)
in_features_box = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes=2)

# Mask head
in_features_mask = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes=2)

model.to(DEVICE)

# ================================
# OPTIMIZER
# ================================
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=1e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# ================================
# TRAINING LOOP
# ================================
OUTPUT_DIR = "./df2_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
writer = SummaryWriter("./runs/df2_logs")

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images = list(img.to(DEVICE) for img in images)
        targets = [{k: v.to(DEVICE) for k,v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    lr_scheduler.step()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}")
    writer.add_scalar("train/loss", avg_loss, epoch)

    # Validation pass
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(img.to(DEVICE) for img in images)
            targets = [{k: v.to(DEVICE) for k,v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            val_loss += sum(loss for loss in loss_dict.values()).item()
    val_loss /= len(val_loader)
    print(f"Validation loss: {val_loss:.4f}")
    writer.add_scalar("val/loss", val_loss, epoch)

    # Save checkpoint
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"model_epoch_{epoch+1}.pth"))

writer.close()
print("🎉 Training complete!")
