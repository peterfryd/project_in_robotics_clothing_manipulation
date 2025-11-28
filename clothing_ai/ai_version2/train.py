import os
import json
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import time

# ================================
# CONFIG
# ================================
DATA_ROOT = "/home/ucloud/Downloads/deepfashion2_original_images"
OUTPUT_DIR = "./df2_output"
CACHE_DIR = "./df2_cache"
TENSORBOARD_DIR = "./runs/df2_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train", "image")
TRAIN_ANN_DIR = os.path.join(DATA_ROOT, "train", "annos")
VAL_IMG_DIR   = os.path.join(DATA_ROOT, "validation", "image")
VAL_ANN_DIR   = os.path.join(DATA_ROOT, "validation", "annos")

BATCH_SIZE = 4
NUM_EPOCHS = 20
NUM_KEYPOINTS = 25
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ================================
# PREPROCESS DATA AND CACHE
# ================================
def preprocess_and_cache(img_dir, ann_dir, split_name):
    cache_split = os.path.join(CACHE_DIR, split_name)
    os.makedirs(cache_split, exist_ok=True)
    files = sorted([f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))])
    cached_files = []
    for img_file in files:
        img_path = os.path.join(img_dir, img_file)
        ann_path = os.path.join(ann_dir, img_file.rsplit(".",1)[0]+".json")
        cache_path = os.path.join(cache_split, img_file.rsplit(".",1)[0]+".pt")

        if os.path.exists(cache_path):
            cached_files.append(cache_path)
            continue

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        with open(ann_path, "r") as f:
            ann = json.load(f)

        obj = ann.get("item1", {})
        if obj.get("category_name") != "short sleeve top":
            obj = ann.get("item2", {})
            if obj.get("category_name") != "short sleeve top":
                # create empty target
                target = {
                    "boxes": torch.zeros((0,4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                    "masks": torch.zeros((0,h,w), dtype=torch.uint8),
                    "keypoints": torch.zeros((0, NUM_KEYPOINTS, 3), dtype=torch.float32)
                }
                torch.save((F.to_tensor(image), target), cache_path)
                cached_files.append(cache_path)
                continue

        # Bounding box
        bbox = torch.tensor([obj["bounding_box"]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)

        # Masks
        segms = obj.get("segmentation", [])
        masks = []
        for seg in segms:
            mask = Image.new("L", (w,h), 0)
            poly = np.array(seg).reshape(-1,2)
            from PIL import ImageDraw
            ImageDraw.Draw(mask).polygon(list(map(tuple, poly)), outline=1, fill=1)
            masks.append(np.array(mask, dtype=np.uint8))
        if masks:
            masks = np.stack(masks)
        else:
            masks = np.zeros((0,h,w), dtype=np.uint8)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Keypoints
        kps = obj.get("landmarks", [])
        if len(kps) != NUM_KEYPOINTS*3:
            kps = (kps + [0]*NUM_KEYPOINTS*3)[:NUM_KEYPOINTS*3]
        kps = torch.tensor(kps, dtype=torch.float32).view(-1, NUM_KEYPOINTS, 3)

        target = {
            "boxes": bbox,
            "labels": labels,
            "masks": masks,
            "keypoints": kps
        }

        torch.save((F.to_tensor(image), target), cache_path)
        cached_files.append(cache_path)
    return cached_files

train_cache_files = preprocess_and_cache(TRAIN_IMG_DIR, TRAIN_ANN_DIR, "train")
val_cache_files = preprocess_and_cache(VAL_IMG_DIR, VAL_ANN_DIR, "val")
print(f"Cached {len(train_cache_files)} training and {len(val_cache_files)} validation samples.")

# ================================
# DATASET
# ================================
class CachedDataset(Dataset):
    def __init__(self, cache_files):
        self.cache_files = cache_files

    def __len__(self):
        return len(self.cache_files)

    def __getitem__(self, idx):
        return torch.load(self.cache_files[idx])

train_dataset = CachedDataset(train_cache_files)
val_dataset = CachedDataset(val_cache_files)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# ================================
# MODEL
# ================================
model = keypointrcnn_resnet50_fpn(weights=None, num_keypoints=NUM_KEYPOINTS)

# Box head (2 classes: background + short sleeve top)
in_features_box = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes=2)

# Mask head (2 classes)
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
# TRAINING LOOP WITH PROGRESS PRINTS
# ================================
writer = SummaryWriter(TENSORBOARD_DIR)

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = list(img.to(DEVICE) for img in images)
        targets = [{k: v.to(DEVICE) for k,v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

        # Print progress every batch
        if (batch_idx+1) % 5 == 0 or (batch_idx+1) == len(train_loader):
            elapsed = time.time() - start_time
            avg_loss = running_loss / (batch_idx+1)
            remaining = elapsed / (batch_idx+1) * (len(train_loader) - (batch_idx+1))
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx+1}/{len(train_loader)}, "
                  f"Loss: {avg_loss:.4f}, ETA: {remaining/60:.1f} min")

    lr_scheduler.step()
    avg_epoch_loss = running_loss / len(train_loader)
    writer.add_scalar("train/loss", avg_epoch_loss, epoch)

    # Validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(img.to(DEVICE) for img in images)
            targets = [{k: v.to(DEVICE) for k,v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            val_loss += sum(loss for loss in loss_dict.values()).item()
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1} validation loss: {val_loss:.4f}")
    writer.add_scalar("val/loss", val_loss, epoch)

    # Save checkpoint
    ckpt_path = os.path.join(OUTPUT_DIR, f"model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"💾 Saved checkpoint: {ckpt_path}")

writer.close()
print("🎉 Training complete!")
