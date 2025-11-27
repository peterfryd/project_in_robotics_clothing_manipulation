#!/usr/bin/env python3
"""
simple_infer.py

Standalone, non-ROS script that loads a ResNet18-based model similar to
`get_landmarks.py`, crops an input image using the same crop logic, runs
inference to predict landmark coordinates, annotates, and saves results.

Edit the `MODEL_PATH` and `IMAGE_PATH` constants below to point to files on
your machine. Run with:

    python3 simple_infer.py

"""
import os
import sys
import cv2
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn as nn
from torchvision import transforms, models

# --------------------------- User-editable paths ---------------------------
# Hardcoded model and image paths — change these to files on your system.
MODEL_PATH = "/home/anders/workspace/project_in_robotics_clothing_manipulation/src/clothing_ai_pkg/data/model.pth"
IMAGE_PATH = "/home/anders/Downloads/images_orientation/5_Color.png"
OUTPUT_DIR = "/home/anders/workspace/project_in_robotics_clothing_manipulation/"
# --------------------------------------------------------------------------

NUM_LANDMARKS = 25
DATA_PER_LANDMARK = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224


def load_model(model_path: str):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_LANDMARKS * DATA_PER_LANDMARK)

    if not os.path.exists(model_path):
        print(f"ERROR: model file not found at {model_path}")
        sys.exit(1)

    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Remove 'backbone.' prefix if present in keys
            if all(k.startswith('backbone.') for k in state_dict.keys()):
                state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            print(f"Loaded model from checkpoint (epoch={checkpoint.get('epoch', 'unknown')})")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded model state_dict from {model_path}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)

    model.to(DEVICE)
    model.eval()
    return model


def run_inference(model, image: np.ndarray):
    # image: OpenCV BGR image (cropped region expected)
    if not isinstance(image, np.ndarray):
        raise RuntimeError("Invalid image type passed to run_inference")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = PILImage.fromarray(rgb)

    tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    img_tensor = tfms(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = model(img_tensor).view(NUM_LANDMARKS, DATA_PER_LANDMARK).cpu()

    # Extract x, y, and visibility
    preds_xy = preds[:, :2]
    visibility = preds[:, 2]

    # Scale from [0-1] to pixel coordinates of the PIL image
    final_preds = preds_xy.clone()
    final_preds[:, 0] *= pil_image.width
    final_preds[:, 1] *= pil_image.height

    # Return as list of [x, y, visibility]
    result = final_preds.numpy().tolist()
    for i, v in enumerate(visibility):
        result[i].append(float(v))
    return result


def annotate_image_and_save(image: np.ndarray, landmarks, out_path: str):
    # Make a copy so original isn't modified by reference
    vis = image.copy()
    for idx, lm in enumerate(landmarks):
        # Handle both 2-value (x, y) and 3-value (x, y, visibility) tuples
        if len(lm) == 3:
            x, y, _ = lm
        else:
            x, y = lm
        
        xi, yi = int(round(x)), int(round(y))
        cv2.circle(vis, (xi, yi), 5, (0, 255, 0), -1)
        cv2.putText(vis, f"({xi},{yi})", (xi + 5, yi - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.putText(vis, f"{idx}", (xi - 10, yi + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)
    print(f"Saved annotated image to {out_path}")


def main():
    print(f"Device: {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Image path: {IMAGE_PATH}")

    model = load_model(MODEL_PATH)

    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: input image not found at {IMAGE_PATH}")
        sys.exit(1)

    cv_image = cv2.imread(IMAGE_PATH)
    if cv_image is None:
        print(f"ERROR: cv2 failed to read image {IMAGE_PATH}")
        sys.exit(1)

    # Crop logic copied from get_landmarks.py
    diff = 1280 - 720
    left_crop = int(1 / 4 * diff)
    right_crop = int(3 / 4 * diff)

    # If the input image isn't exactly 1280x720, adapt gracefully by
    # computing the center crop of width 720.
    h, w = cv_image.shape[:2]
    if w >= 1280 and h >= 720:
        cropped = cv_image[:, left_crop:1280 - right_crop]
    else:
        # Center-crop to square 720x720 if possible
        crop_w = min(720, w)
        crop_h = min(720, h)
        start_x = max(0, (w - crop_w) // 2)
        start_y = max(0, (h - crop_h) // 2)
        cropped = cv_image[start_y:start_y + crop_h, start_x:start_x + crop_w]

    # Rotate 180 degrees like original script
    cropped_rot = cv2.rotate(cropped, cv2.ROTATE_180)

    # Run inference on rotated crop
    preds = run_inference(model, cropped_rot)

    # Convert predictions back to cropped image coordinates (reverse rotation)
    crop_h, crop_w = cropped_rot.shape[:2]
    corrected = []
    for pred in preds:
        # Unpack x, y, and optionally visibility
        if len(pred) == 3:
            x, y, vis = pred
        else:
            x, y = pred
            vis = None
        
        x_c = crop_w - x
        y_c = crop_h - y

        # Move slightly toward center (same heuristic as original)
        point = np.array([x_c, y_c])
        center = np.array([crop_w / 2.0, crop_h / 2.0])
        vec = (center - point)
        norm = np.linalg.norm(vec)
        if norm > 0:
            new_point = point + vec / norm * 60
        else:
            new_point = point

        # If original cropping used left_crop offset, add it back. Here we
        # only know the simple case (w>=1280) where left_crop was used.
        if w >= 1280 and h >= 720:
            x_orig = new_point[0] + left_crop
        else:
            # approximate original coordinates relative to the original image
            # using the center-crop offset
            start_x = max(0, (w - min(720, w)) // 2)
            x_orig = new_point[0] + start_x
        y_orig = new_point[1]

        if vis is not None:
            corrected.append((float(x_orig), float(y_orig), vis))
        else:
            corrected.append((float(x_orig), float(y_orig)))

    # Save annotated images
    base_out = os.path.join(OUTPUT_DIR, 'simple_infer')
    os.makedirs(base_out, exist_ok=True)
    annotate_image_and_save(cropped_rot, preds, os.path.join(base_out, 'cropped_rotated_landmarks.png'))
    # annotate on original image using corrected coordinates
    annotate_image_and_save(cv_image, corrected, os.path.join(base_out, 'original_with_corrected_landmarks.png'))

    print("\nPredicted landmarks (first 8 shown):")
    for i, pred in enumerate(corrected[:8]):
        if len(pred) == 3:
            x, y, vis = pred
            print(f"  {i}: x={x:.1f}, y={y:.1f}, visibility={vis:.3f}")
        else:
            x, y = pred
            print(f"  {i}: x={x:.1f}, y={y:.1f}")


if __name__ == '__main__':
    main()
