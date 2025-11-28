import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import cv2

# ================================
# CONFIG
# ================================
DATA_ROOT = "/home/ucloud/Downloads/deepfashion2_original_images"
OUTPUT_DIR = "./df2_output"
TENSORBOARD_DIR = "./runs/df2_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train", "image")
TRAIN_ANN_DIR = os.path.join(DATA_ROOT, "train", "annos")
VAL_IMG_DIR   = os.path.join(DATA_ROOT, "validation", "image")
VAL_ANN_DIR   = os.path.join(DATA_ROOT, "validation", "annos")

BATCH_SIZE = 4
NUM_WORKERS = 4
LR = 1e-4
EPOCHS = 20
CHECKPOINT_EVERY = 1000

# ================================
# UTILITY: Category mapping
# ================================
def build_category_mapping(ann_dir):
    cat_ids = set()
    for f in os.listdir(ann_dir):
        if not f.endswith(".json"):
            continue
        ann = json.load(open(os.path.join(ann_dir, f)))
        for key in ["item1", "item2"]:
            if key in ann:
                cat_ids.add(ann[key]["category_id"])
    cat_ids = sorted(list(cat_ids))
    cat_id_map = {cid: i for i, cid in enumerate(cat_ids)}
    return cat_id_map, len(cat_ids)

CATEGORY_MAP, NUM_CLASSES = build_category_mapping(TRAIN_ANN_DIR)

# ================================
# DATASET
# ================================
class DeepFashion2Dataset(Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith((".jpg",".png"))])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(self.ann_dir, img_name.rsplit(".",1)[0]+".json")
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        with open(ann_path, "r") as f:
            ann = json.load(f)

        boxes = []
        labels = []
        masks = []
        keypoints = []

        for key in ["item1","item2"]:
            if key not in ann:
                continue
            item = ann[key]
            if "bounding_box" not in item or "category_id" not in item:
                continue
            x0,y0,x1,y1 = item["bounding_box"]
            boxes.append([x0, y0, x1, y1])
            labels.append(CATEGORY_MAP[item["category_id"]]+1)  # +1 because background=0
            # Segmentation masks
            mask = np.zeros((height, width), dtype=np.uint8)
            for poly in item.get("segmentation", []):
                pts = np.array(poly, dtype=np.int32).reshape(-1,2)
                cv2.fillPoly(mask, [pts], 1)
            masks.append(mask)
            # Keypoints
            lm = item.get("landmarks", [])
            if len(lm) % 3 != 0:
                lm = lm + [0]*(3 - len(lm)%3)
            kp = []
            for i in range(0,len(lm),3):
                x, y, v = lm[i:i+3]
                kp.append([x, y, v])
            keypoints.append(kp)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["keypoints"] = keypoints
        target["image_id"] = torch.tensor([idx])

        if self.transforms:
            img = self.transforms(img)

        return img, target

# ================================
# TRANSFORMS
# ================================
train_tfms = transforms.Compose([
    transforms.ToTensor(),
])

val_tfms = transforms.Compose([
    transforms.ToTensor(),
])

# ================================
# DATALOADERS
# ================================
train_dataset = DeepFashion2Dataset(TRAIN_IMG_DIR, TRAIN_ANN_DIR, transforms=train_tfms)
val_dataset   = DeepFashion2Dataset(VAL_IMG_DIR, VAL_ANN_DIR, transforms=val_tfms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=lambda x: tuple(zip(*x)))
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=lambda x: tuple(zip(*x)))

# ================================
# MODEL
# ================================
def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)
    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes+1)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(NUM_CLASSES).to(device)

# ================================
# OPTIMIZER
# ================================
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(params, lr=LR)

# ================================
# TRAINING LOOP
# ================================
writer = SummaryWriter(TENSORBOARD_DIR)
global_step = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, targets in train_loader:
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        writer.add_scalar("train/loss", losses.item(), global_step)
        global_step += 1

        if global_step % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(OUTPUT_DIR, f"model_step_{global_step}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

    print(f"Epoch {epoch+1} | Avg Loss: {running_loss/len(train_loader):.4f}")

    # ================================
    # VALIDATION LOOP (every epoch)
    # ================================
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
    avg_val_loss = val_loss / len(val_loader)
    writer.add_scalar("val/loss", avg_val_loss, epoch)
    print(f"Epoch {epoch+1} | Validation Loss: {avg_val_loss:.4f}")

# ================================
# FINAL SAVE
# ================================
final_path = os.path.join(OUTPUT_DIR, "model_final.pth")
torch.save(model.state_dict(), final_path)
writer.close()
print(f"🎉 Training complete! Model saved at {final_path}")
