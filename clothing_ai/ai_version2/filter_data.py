# filter_data.py
import os
import json
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F
from tqdm import tqdm

NUM_KEYPOINTS = 25
DATA_ROOT = "/home/ucloud/Downloads/deepfashion2_original_images"
CACHE_DIR = "./df2_cache"

TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train", "image")
TRAIN_ANN_DIR = os.path.join(DATA_ROOT, "train", "annos")
VAL_IMG_DIR   = os.path.join(DATA_ROOT, "validation", "image")
VAL_ANN_DIR   = os.path.join(DATA_ROOT, "validation", "annos")

def process_split(img_dir, ann_dir, split_name):
    split_cache = os.path.join(CACHE_DIR, split_name)
    os.makedirs(split_cache, exist_ok=True)

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))])
    cached_files = []

    for img_file in tqdm(img_files, desc=f"Processing {split_name}"):
        img_path = os.path.join(img_dir, img_file)
        ann_path = os.path.join(ann_dir, img_file.rsplit(".", 1)[0] + ".json")
        cache_path = os.path.join(split_cache, img_file.rsplit(".", 1)[0] + ".pt")

        # Skip if already cached
        if os.path.exists(cache_path):
            cached_files.append(cache_path)
            continue

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        with open(ann_path, "r") as f:
            ann = json.load(f)

        # Filter short sleeve tops
        obj = ann.get("item1", {})
        if obj.get("category_name") != "short sleeve top":
            obj = ann.get("item2", {})
            if obj.get("category_name") != "short sleeve top":
                continue  # skip this image entirely

        # Bounding box
        bbox = torch.tensor([obj["bounding_box"]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)

        # Masks
        segms = obj.get("segmentation", [])
        masks = []
        for seg in segms:
            mask = Image.new("L", (w, h), 0)
            poly = np.array(seg).reshape(-1, 2)
            from PIL import ImageDraw
            ImageDraw.Draw(mask).polygon(list(map(tuple, poly)), outline=1, fill=1)
            masks.append(np.array(mask, dtype=np.uint8))
        if masks:
            masks = np.stack(masks)
        else:
            masks = np.zeros((0, h, w), dtype=np.uint8)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Keypoints
        kps = obj.get("landmarks", [])
        if len(kps) != NUM_KEYPOINTS * 3:
            kps = (kps + [0] * NUM_KEYPOINTS * 3)[:NUM_KEYPOINTS * 3]
        kps = torch.tensor(kps, dtype=torch.float32).view(-1, NUM_KEYPOINTS, 3)

        target = {
            "boxes": bbox,
            "labels": labels,
            "masks": masks,
            "keypoints": kps
        }

        torch.save((F.to_tensor(image), target), cache_path)
        cached_files.append(cache_path)

    print(f"✅ Cached {len(cached_files)} samples for split '{split_name}'")
    return cached_files

if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    train_files = process_split(TRAIN_IMG_DIR, TRAIN_ANN_DIR, "train")
    val_files = process_split(VAL_IMG_DIR, VAL_ANN_DIR, "val")
    print(f"Preprocessing complete. Train: {len(train_files)}, Val: {len(val_files)}")
