import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, KeypointRCNNPredictor
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

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

NUM_KEYPOINTS = 25  # short sleeve top landmarks
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ================================
# DATASET
# ================================
class DeepFashion2Dataset(Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.files = sorted([f for f in os.listdir(img_dir) if f.endswith((".jpg",".png"))])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_file = self.files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        ann_path = os.path.join(self.ann_dir, img_file.rsplit(".",1)[0] + ".json")
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        with open(ann_path, "r") as f:
            ann = json.load(f)

        # only keep short sleeve tops
        item = ann.get("item1", None)
        if item is None or item.get("category_name") != "short sleeve top":
            # skip non-short-top images
            return self.__getitem__((idx+1) % len(self.files))

        # Bounding box
        bbox = torch.tensor(item["bounding_box"], dtype=torch.float32)
        # Convert [x1,y1,x2,y2] -> [xmin,ymin,xmax,ymax]
        bbox = bbox.unsqueeze(0)

        # Segmentation masks
        masks = []
        for seg in item.get("segmentation", []):
            seg_np = np.array(seg, dtype=np.int32).reshape(-1,2)
            mask = np.zeros((h, w), dtype=np.uint8)
            import cv2
            cv2.fillPoly(mask, [seg_np], 1)
            masks.append(mask)
        if len(masks) == 0:
            masks = torch.zeros((1,h,w), dtype=torch.uint8)
        else:
            masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)

        # Keypoints (25 landmarks)
        kpts_raw = item.get("landmarks", [])
        kpts = []
        for i in range(0, len(kpts_raw), 3):
            kpts.append([kpts_raw[i], kpts_raw[i+1], kpts_raw[i+2]])
        kpts = torch.as_tensor([kpts], dtype=torch.float32)  # [num_objects, num_keypoints, 3]

        target = {}
        target["boxes"] = bbox
        target["labels"] = torch.tensor([0], dtype=torch.int64)  # single class
        target["masks"] = masks
        target["keypoints"] = kpts
        target["image_id"] = torch.tensor([idx])

        if self.transforms:
            img = self.transforms(img)

        return img, target

# ================================
# TRANSFORMS
# ================================
def get_transform(train=True):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

# ================================
# MODEL
# ================================
def get_model(num_classes=1, num_keypoints=25):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT", weights_backbone="DEFAULT")
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = MaskRCNNPredictor(in_features, num_classes)

    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes)

    # Replace keypoint predictor
    in_features_kpt = model.roi_heads.keypoint_predictor.kp_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_features_kpt, hidden_layer, num_keypoints)

    return model

# ================================
# DATA LOADERS
# ================================
train_dataset = DeepFashion2Dataset(TRAIN_IMG_DIR, TRAIN_ANN_DIR, transforms=get_transform())
val_dataset   = DeepFashion2Dataset(VAL_IMG_DIR, VAL_ANN_DIR, transforms=get_transform())

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader   = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# ================================
# TRAINING SETUP
# ================================
model = get_model().to(DEVICE)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=1e-4)
writer = SummaryWriter(TENSORBOARD_DIR)

num_epochs = 10
checkpoint_interval = 2000
global_step = 0

for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = list(img.to(DEVICE) for img in images)
        targets = [{k: v.to(DEVICE) for k,v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Tensorboard logging
        if global_step % 20 == 0:
            writer.add_scalar("train/loss_total", losses.item(), global_step)
            for k,v in loss_dict.items():
                writer.add_scalar(f"train/{k}", v.item(), global_step)

        # Save checkpoint
        if global_step % checkpoint_interval == 0 and global_step > 0:
            ckpt_path = os.path.join(OUTPUT_DIR, f"model_step_{global_step}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

        global_step += 1

    # Validation loss (once per epoch)
    model.eval()
    val_losses = []
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(img.to(DEVICE) for img in images)
            targets = [{k: v.to(DEVICE) for k,v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_losses.append(losses.item())
    avg_val_loss = np.mean(val_losses)
    writer.add_scalar("val/loss_total", avg_val_loss, epoch)
    print(f"Epoch {epoch} - Avg Validation Loss: {avg_val_loss:.4f}")

writer.close()
print("🎉 Training complete!")
