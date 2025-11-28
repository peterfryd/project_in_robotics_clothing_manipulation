import os
import json
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, KeypointRCNNPredictor
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T

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

NUM_CLASSES = 1 + 1  # 1 short top + background
NUM_KEYPOINTS = 25   # short top landmarks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NUM_WORKERS = 4
LR = 1e-4
MAX_EPOCHS = 30
CHECKPOINT_EVERY = 2000
VAL_EVERY = 5000

# ================================
# DATASET
# ================================
class DeepFashion2Dataset(Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.files = [f for f in os.listdir(img_dir) if f.endswith((".jpg",".png"))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(self.ann_dir, img_name.rsplit(".", 1)[0] + ".json")

        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        # Load annotations
        with open(ann_path, "r") as f:
            ann = json.load(f)

        # Filter only short sleeve tops
        records = []
        for key in ["item1", "item2"]:
            if key not in ann:
                continue
            item = ann[key]
            if item.get("category_name") != "short sleeve top":
                continue
            # Skip if no bounding box
            if "bounding_box" not in item:
                continue
            bbox = item["bounding_box"]
            # Convert bbox to [x_min, y_min, x_max, y_max]
            bbox = torch.tensor(bbox, dtype=torch.float32).unsqueeze(0)

            # Mask
            masks = []
            for seg in item.get("segmentation", []):
                mask = Image.new("L", (width, height), 0)
                xy = np.array(seg).reshape((-1, 2))
                Image.Draw.Draw(mask).polygon([tuple(p) for p in xy], outline=1, fill=1)
                masks.append(np.array(mask))
            if masks:
                masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            else:
                masks = torch.zeros((0, height, width), dtype=torch.uint8)

            # Keypoints
            keypoints = item.get("landmarks", [])
            if len(keypoints) == NUM_KEYPOINTS*3:
                keypoints = torch.tensor(np.array(keypoints).reshape(-1,3), dtype=torch.float32)
            else:
                # Skip if keypoints are incomplete
                continue

            target = {
                "boxes": bbox,
                "labels": torch.tensor([1], dtype=torch.int64),
                "masks": masks,
                "keypoints": keypoints,
            }
            records.append(target)

        if len(records) == 0:
            # dummy empty target
            target = {
                "boxes": torch.zeros((0,4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0,height,width), dtype=torch.uint8),
                "keypoints": torch.zeros((0, NUM_KEYPOINTS, 3), dtype=torch.float32)
            }
        else:
            # concatenate targets if multiple items in image
            target = {}
            for k in ["boxes","labels","masks","keypoints"]:
                target[k] = torch.cat([r[k] for r in records], dim=0)

        if self.transforms:
            image = self.transforms(image)

        return image, target

# ================================
# TRANSFORMS
# ================================
def get_transform():
    return T.Compose([
        T.Resize((800, 800)),
        T.ToTensor(),
    ])

# ================================
# MODEL
# ================================
def get_model():
    # Pretrained backbone only
    model = maskrcnn_resnet50_fpn(weights_backbone=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT, weights=None)

    # Box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    # Mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES)

    # Keypoint predictor
    in_features_keypoint = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_features_keypoint, hidden_layer, NUM_KEYPOINTS)

    return model

# ================================
# TRAINING LOOP
# ================================
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def train():
    # Dataset
    train_dataset = DeepFashion2Dataset(TRAIN_IMG_DIR, TRAIN_ANN_DIR, transforms=get_transform())
    val_dataset = DeepFashion2Dataset(VAL_IMG_DIR, VAL_ANN_DIR, transforms=get_transform())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = get_model().to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR)

    writer = SummaryWriter(TENSORBOARD_DIR)
    step = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for images, targets in train_loader:
            images = list(img.to(DEVICE) for img in images)
            targets = [{k:v.to(DEVICE) for k,v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if step % 20 == 0:
                writer.add_scalar("train/loss", losses.item(), step)
                print(f"Epoch {epoch}, Step {step}, Loss: {losses.item():.4f}")

            if step % CHECKPOINT_EVERY == 0 and step > 0:
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"model_step_{step}.pth"))
                print(f"💾 Saved checkpoint at step {step}")

            if step % VAL_EVERY == 0 and step > 0:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for v_images, v_targets in val_loader:
                        v_images = list(img.to(DEVICE) for img in v_images)
                        v_targets = [{k:v.to(DEVICE) for k,v in t.items()} for t in v_targets]
                        v_loss_dict = model(v_images, v_targets)
                        val_loss += sum(l for l in v_loss_dict.values()).item()
                val_loss /= len(val_loader)
                writer.add_scalar("val/loss", val_loss, step)
                print(f"📝 Validation loss at step {step}: {val_loss:.4f}")
                model.train()

            step += 1

    # Final save
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_final.pth"))
    writer.close()
    print("🎉 Training complete!")

if __name__ == "__main__":
    train()
