import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# ==== CONFIG ====
DATA_ROOT = "/home/peter/uni/clothing_ai/Data"
CKPT_PATH = "./checkpoints/model_final.pth"  # change if you want to validate earlier checkpoint
IMG_SIZE = 224
BATCH_SIZE = 16

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

        landmarks = torch.tensor(anno["landmarks"], dtype=torch.float32)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        landmarks[:, 0] /= w
        landmarks[:, 1] /= h

        if self.transform:
            img = self.transform(img)

        landmarks = landmarks.flatten()
        return img, landmarks


# ==== MODEL ====
class LandmarkRegressor(nn.Module):
    def __init__(self, num_landmarks):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_landmarks * 3)

    def forward(self, x):
        return self.backbone(x)


# ==== LOAD DATA ====
val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

val_dataset = ShortSleeveLandmarkDataset(os.path.join(DATA_ROOT, "validation"), transform=val_tfms)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ==== INIT MODEL ====
first_ann = json.load(open(os.path.join(DATA_ROOT, "validation", "annotations",
                        os.listdir(os.path.join(DATA_ROOT, "validation", "annotations"))[0])))
num_landmarks = len(first_ann["landmarks"])

model = LandmarkRegressor(num_landmarks=num_landmarks)
model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ==== VALIDATION LOOP ====
criterion = nn.MSELoss()
val_loss = 0.0

with torch.no_grad():
    for imgs, landmarks in tqdm(val_loader, desc="Validating"):
        imgs, landmarks = imgs.to(device), landmarks.to(device)
        preds = model(imgs)
        val_loss += criterion(preds, landmarks).item()

val_loss /= len(val_loader)
print(f"✅ Validation MSE Loss: {val_loss:.4f}")
