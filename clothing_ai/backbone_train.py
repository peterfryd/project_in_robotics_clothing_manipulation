import os
import json
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import keypointrcnn_resnet50_fpn
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# =============================
# CONFIG
# =============================
DATA_ROOT = "/home/ucloud/Downloads/deepfashion2_original_images"
OUTPUT_DIR = "./df2_output_landmarks"
TENSORBOARD_DIR = "./runs/df2_landmark_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train", "image")
TRAIN_ANN_DIR = os.path.join(DATA_ROOT, "train", "annos")
VAL_IMG_DIR   = os.path.join(DATA_ROOT, "validation", "image")
VAL_ANN_DIR   = os.path.join(DATA_ROOT, "validation", "annos")

NUM_KEYPOINTS = 25  # landmarks per short sleeve top
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 2
NUM_WORKERS = 4
LR = 1e-4
NUM_EPOCHS = 50
CHECKPOINT_EVERY = 2000

# =============================
# DATASET
# =============================
class ShortTopKeypointDataset(Dataset):
    def __init__(self, img_dir, ann_dir):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith((".jpg",".png"))])
        self.img_files = [f for f in self.img_files if self._has_short_top(os.path.join(ann_dir, f.rsplit(".",1)[0]+".json"))]

    def _has_short_top(self, ann_path):
        if not os.path.exists(ann_path):
            return False
        ann = json.load(open(ann_path))
        for key in ["item1","item2"]:
            if key in ann and ann[key]["category_name"] == "short sleeve top":
                return True
        return False

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        ann_path = os.path.join(self.ann_dir, img_file.rsplit(".",1)[0]+".json")

        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        ann = json.load(open(ann_path))

        boxes = []
        keypoints = []

        for key in ["item1","item2"]:
            if key not in ann:
                continue
            item = ann[key]
            if item["category_name"] != "short sleeve top":
                continue
            boxes.append(item["bounding_box"])

            kps = item.get("landmarks", [])
            kps_formatted = []
            for i in range(0, len(kps), 3):
                x, y, v = kps[i:i+3]
                if x==0 and y==0:
                    v = 0
                kps_formatted.append([x,y,v])
            if len(kps_formatted) < NUM_KEYPOINTS:
                for _ in range(NUM_KEYPOINTS - len(kps_formatted)):
                    kps_formatted.append([0,0,0])
            keypoints.append(kps_formatted[:NUM_KEYPOINTS])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "keypoints": keypoints}
        img = torch.as_tensor(img.transpose(2,0,1), dtype=torch.float32) / 255.0
        return img, target

# =============================
# DATA LOADERS
# =============================
def collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)

train_dataset = ShortTopKeypointDataset(TRAIN_IMG_DIR, TRAIN_ANN_DIR)
val_dataset   = ShortTopKeypointDataset(VAL_IMG_DIR, VAL_ANN_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False,
                          num_workers=NUM_WORKERS, collate_fn=collate_fn)

# =============================
# MODEL
# =============================
model = keypointrcnn_resnet50_fpn(weights="DEFAULT", num_keypoints=NUM_KEYPOINTS)
model.to(DEVICE)

# =============================
# OPTIMIZER
# =============================
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)

# =============================
# TRAIN LOOP
# =============================
writer = SummaryWriter(TENSORBOARD_DIR)
global_step = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    for imgs, targets in train_loader:
        imgs = [img.to(DEVICE) for img in imgs]
        targets = [{k:v.to(DEVICE) for k,v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if global_step % 20 == 0:
            for k,v in loss_dict.items():
                writer.add_scalar(f"train/{k}", v.item(), global_step)
            writer.add_scalar("train/total_loss", losses.item(), global_step)
            print(f"Epoch {epoch} Step {global_step} Loss: {losses.item():.4f}")

        if global_step % CHECKPOINT_EVERY == 0 and global_step > 0:
            ckpt_path = os.path.join(OUTPUT_DIR, f"model_step_{global_step}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

        global_step += 1

    # validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = [img.to(DEVICE) for img in imgs]
            targets = [{k:v.to(DEVICE) for k,v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            val_losses.append(sum(loss for loss in loss_dict.values()).item())
    avg_val_loss = np.mean(val_losses)
    writer.add_scalar("val/loss", avg_val_loss, epoch)
    print(f"Epoch {epoch} Validation Loss: {avg_val_loss:.4f}")

writer.close()
print("🎉 Training complete!")
