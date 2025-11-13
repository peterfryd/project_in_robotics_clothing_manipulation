import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# ==== CONFIG ====
PRETRAINED_PATH = "./checkpoints/model_backbone.pth"

# --- Training Data ---
NEW_ANNOS_DIR = "./data/augmented_annos"
NEW_IMAGES_DIR = "./data/augmented_images"

# --- Validation Data ---
VAL_ANNOS_DIR = "./data/val_annos"
VAL_IMAGES_DIR = "./data/val_images"

# --- Save/Log Paths ---
SAVE_DIR = "./checkpoints"
LOG_DIR = "./runs/head_logs_deep" # New log dir for this experiment
MODEL_SAVE_PREFIX = "head_model_deep" # New model name

# --- HPC HYPERPARAMETERS ---
LR = 1e-3           # Initial LR for the head (will be 10 times smaller after unfreezing backend)
BACKBONE_LR = 1e-5  # Very low LR for fine-tuning
EPOCHS = 300        # More time for fine-tuning
UNFREEZE_EPOCH = 15 # Unfreeze after head is stable
BATCH_SIZE = 32     # Larger batch for HPC GPU
NUM_WORKERS = 8     # Use more CPUs for data loading
NUM_NEW_LANDMARKS = 8
IMG_SIZE = 224
SAVE_EVERY_EPOCH = 30
WEIGHT_DECAY = 1e-4 # Stronger regularization
# --- END HYPERPARAMETERS ---

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print(f"Batch Size: {BATCH_SIZE} | Workers: {NUM_WORKERS} | Unfreeze at: {UNFREEZE_EPOCH}")
writer = SummaryWriter(LOG_DIR)

# ==== MASKED L1 LOSS ====
def masked_l1_loss(preds, targets):
    vis_mask = (targets[:, :, 2] > 0).unsqueeze(-1).float() 
    diff = torch.abs(preds - targets[:, :, :2])
    loss = (diff * vis_mask).sum() / vis_mask.sum().clamp(min=1)
    return loss

# ==== DATASET ====
class NewSmallDataset(Dataset):
    def __init__(self, annos_dir, images_dir, transform=None):
        self.transform = transform
        self.samples = []
        if not os.path.exists(annos_dir):
             raise FileNotFoundError(f"Annotations directory not found: {annos_dir}")
        for jf in sorted([f for f in os.listdir(annos_dir) if f.endswith('.json')]):
            with open(os.path.join(annos_dir, jf), 'r') as f:
                data = json.load(f)
            self.samples.append({
                'img_path': os.path.join(images_dir, data['image']),
                'landmarks': data['landmarks']
            })
        print(f"✅ Loaded {len(self.samples)} samples from {annos_dir}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
            item = self.samples[idx]
            try:
                img = Image.open(item['img_path']).convert("RGB")
            except FileNotFoundError:
                raise FileNotFoundError(f"Image missing: {item['img_path']}")
            landmarks = torch.tensor(item['landmarks'], dtype=torch.float32)
            landmarks[:, 0] /= 100.0
            landmarks[:, 1] /= 100.0
            if self.transform:
                img = self.transform(img)
            return img, landmarks

# ==== MODEL SETUP (Updated with Deeper Head) ====
print(f"🔄 Loading backbone from {PRETRAINED_PATH}...")
base_model = models.resnet18(weights=None)
checkpoint = torch.load(PRETRAINED_PATH, map_location=device)
current_state = base_model.state_dict()
pretrained_backbone = {k: v for k, v in checkpoint.items() if k in current_state and 'fc' not in k}
current_state.update(pretrained_backbone)
base_model.load_state_dict(current_state)

for param in base_model.parameters():
    param.requires_grad = False

# --- DEEPER HEAD BLOCK ---
in_feats = base_model.fc.in_features
hidden_dim = 256 # A small intermediate layer

base_model.fc = nn.Sequential(
    nn.Linear(in_feats, hidden_dim), # 512 -> 256
    nn.ReLU(),                       # Activation
    nn.Dropout(p=0.5),               # Regularization
    nn.Linear(hidden_dim, NUM_NEW_LANDMARKS * 2) # 256 -> 16
)
# --- END DEEPER HEAD ---

model = base_model.to(device)

# ==== DATALOADERS ====
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
])
train_dataset = NewSmallDataset(NEW_ANNOS_DIR, NEW_IMAGES_DIR, transform=train_tfms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
val_dataset = NewSmallDataset(VAL_ANNOS_DIR, VAL_IMAGES_DIR, transform=val_tfms)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# --- Optimizer (Initially only trains head) ---
optimizer = optim.Adam(model.fc.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ==== TRAINING LOOP ====
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    
    # --- UNFREEZE LOGIC ---
    if epoch == UNFREEZE_EPOCH:
        print(f"🔓 Unfreezing backbone at epoch {epoch}...")
        for param in model.parameters():
            param.requires_grad = True
        
        backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]
        head_params = model.fc.parameters()
        
        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': BACKBONE_LR}, # 1e-5
            {'params': head_params, 'lr': LR * 0.1}         # 1e-4
        ], weight_decay=WEIGHT_DECAY)

    # --- 1. TRAINING ---
    model.train()
    running_train_loss = 0.0
    for imgs, landmarks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False):
        imgs = imgs.to(device)
        landmarks = landmarks.to(device)
        
        preds = model(imgs).view(-1, NUM_NEW_LANDMARKS, 2)
        loss = masked_l1_loss(preds, landmarks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)

    # --- 2. VALIDATION ---
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for imgs, landmarks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False):
            imgs = imgs.to(device)
            landmarks = landmarks.to(device)
            
            preds = model(imgs).view(-1, NUM_NEW_LANDMARKS, 2)
            loss = masked_l1_loss(preds, landmarks)
            running_val_loss += loss.item()
            
    avg_val_loss = running_val_loss / len(val_loader)

    # --- 3. LOGGING & SAVING ---
    
    # Log to TensorBoard
    writer.add_scalars("Loss", 
                       {'Train': avg_train_loss, 'Validation': avg_val_loss}, 
                       epoch)

    print(f"Epoch {epoch+1:02d} | Train L1 Loss: {avg_train_loss:.4f} | Val L1 Loss: {avg_val_loss:.4f}")

    # Save Best Model (based on validation loss)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{MODEL_SAVE_PREFIX}_best.pth"))
        print(f"🏆 New best model saved with val loss: {best_val_loss:.4f}")

    if (epoch + 1) % SAVE_EVERY_EPOCH == 0:
        ckpt_path = os.path.join(SAVE_DIR, f"{MODEL_SAVE_PREFIX}_ep{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")

torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{MODEL_SAVE_PREFIX}_final.pth"))
writer.close()
print("🎉 Done! Best validation loss: {:.4f}".format(best_val_loss))