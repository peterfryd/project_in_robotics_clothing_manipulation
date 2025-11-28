import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.ops import box_iou
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

# ================================
# CONFIG
# ================================
DATA_ROOT = "/home/ucloud/deepfashion2_original_images"
OUTPUT_DIR = "./df2_output"
TENSORBOARD_DIR = "./runs/df2_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train", "image")
TRAIN_ANN_DIR = os.path.join(DATA_ROOT, "train", "annos")
VAL_IMG_DIR   = os.path.join(DATA_ROOT, "validation", "image")
VAL_ANN_DIR   = os.path.join(DATA_ROOT, "validation", "annos")

# Only short sleeve tops
SHORT_TOP_CATEGORY_NAME = "short sleeve top"
NUM_KEYPOINTS = 25  # fixed

# ================================
# DATASET
# ================================
class DeepFashion2ShortTopDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith((".jpg",".png"))])
        self.data = self._load_annotations()

    def _load_annotations(self):
        dataset = []
        for idx, img_file in enumerate(self.img_files):
            ann_path = os.path.join(self.ann_dir, img_file.rsplit(".",1)[0] + ".json")
            if not os.path.exists(ann_path):
                continue
            with open(ann_path, "r") as f:
                ann = json.load(f)

            for key in ["item1","item2"]:
                if key not in ann:
                    continue
                item = ann[key]
                if item.get("category_name","") != SHORT_TOP_CATEGORY_NAME:
                    continue
                # check keypoints length
                if "landmarks" not in item or len(item["landmarks"]) != NUM_KEYPOINTS * 3:
                    continue
                dataset.append({
                    "img_file": img_file,
                    "annotation": item
                })
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        img_path = os.path.join(self.img_dir, record["img_file"])
        ann = record["annotation"]

        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        # Prepare target
        boxes = torch.as_tensor([ann["bounding_box"]], dtype=torch.float32).unsqueeze(0)  # [1,4]
        # Convert segmentation into list of masks
        masks = []
        for poly in ann.get("segmentation", []):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            if len(poly) < 6:
                continue
            pts = np.array(poly).reshape(-1,2).astype(np.int32)
            import cv2
            cv2.fillPoly(mask, [pts], 1)
            masks.append(mask)
        if len(masks) == 0:
            masks = np.zeros(image.shape[:2], dtype=np.uint8)
            masks = torch.as_tensor(masks[None], dtype=torch.uint8)
        else:
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)

        keypoints = np.array(ann["landmarks"]).reshape(-1,3)
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32)

        target = {
            "boxes": torch.as_tensor([ann["bounding_box"]], dtype=torch.float32),
            "labels": torch.ones((1,), dtype=torch.int64),  # only one class
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "keypoints": keypoints
        }

        if self.transforms:
            image = self.transforms(image)

        image = torch.as_tensor(image.transpose(2,0,1), dtype=torch.float32)
        return image, target

# Simple resizing transform
def get_transform():
    return T.Compose([
        T.ToTensor(),
    ])

# ================================
# DATALOADERS
# ================================
train_dataset = DeepFashion2ShortTopDataset(TRAIN_IMG_DIR, TRAIN_ANN_DIR, transforms=get_transform())
val_dataset   = DeepFashion2ShortTopDataset(VAL_IMG_DIR, VAL_ANN_DIR, transforms=get_transform())

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))

# ================================
# MODEL
# ================================
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

NUM_CLASSES = 1  # only short top + background
model = maskrcnn_resnet50_fpn(
    weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
    num_classes=NUM_CLASSES + 1,  # include background
    num_keypoints=NUM_KEYPOINTS
)
model.to(device)

# ================================
# OPTIMIZER
# ================================
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=1e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)

# ================================
# TENSORBOARD
# ================================
writer = SummaryWriter(TENSORBOARD_DIR)

# ================================
# TRAIN LOOP
# ================================
NUM_EPOCHS = 10
SAVE_EVERY = 1000
VAL_EVERY  = 500

global_step = 0
for epoch in range(NUM_EPOCHS):
    model.train()
    for images, targets in train_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if global_step % 10 == 0:
            for k, v in loss_dict.items():
                writer.add_scalar(f"train/{k}", v.item(), global_step)
            writer.add_scalar("train/total_loss", losses.item(), global_step)

        # Validation
        if global_step % VAL_EVERY == 0 and global_step > 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for val_images, val_targets in val_loader:
                    val_images = list(img.to(device) for img in val_images)
                    val_targets = [{k: v.to(device) for k,v in t.items()} for t in val_targets]
                    loss_dict_val = model(val_images, val_targets)
                    val_loss = sum(loss for loss in loss_dict_val.values())
                    val_losses.append(val_loss.item())
            avg_val_loss = np.mean(val_losses)
            writer.add_scalar("val/total_loss", avg_val_loss, global_step)
            print(f"[Epoch {epoch}] Step {global_step} | Val Loss: {avg_val_loss:.4f}")
            model.train()

        # Save checkpoint
        if global_step % SAVE_EVERY == 0 and global_step > 0:
            checkpoint_path = os.path.join(OUTPUT_DIR, f"model_step_{global_step}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")

        global_step += 1
    lr_scheduler.step()

writer.close()
print("🎉 Training complete!")
