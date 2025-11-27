#!/usr/bin/env python3
"""
analyze_visibility.py

Batch process all images in a folder, compute landmark visibility scores,
and generate visualizations showing:
  - Mean visibility per image
  - Variance of visibility per image
  - All individual visibility scores across all landmarks in all images

Edit the INPUT_FOLDER below to point to your image directory.
Run with:

    python3 analyze_visibility.py

"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image as PILImage
import torch
import torch.nn as nn
from torchvision import transforms, models
import matplotlib.pyplot as plt

# --------------------------- User-editable paths ---------------------------
INPUT_FOLDER = "/home/anders/Downloads/images_orientation"
MODEL_PATH = "/home/anders/workspace/project_in_robotics_clothing_manipulation/src/clothing_ai_pkg/data/model.pth"
OUTPUT_DIR = "/home/anders/workspace/project_in_robotics_clothing_manipulation/visibility_analysis"
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
    """Run inference and return [x, y, visibility] for each landmark."""
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

    preds_xy = preds[:, :2]
    visibility = preds[:, 2]

    final_preds = preds_xy.clone()
    final_preds[:, 0] *= pil_image.width
    final_preds[:, 1] *= pil_image.height

    result = final_preds.numpy().tolist()
    for i, v in enumerate(visibility):
        result[i].append(float(v))
    return result


def process_image(model, image_path: str):
    """Process a single image and return landmark predictions, original image, and crop/rotation info."""
    if not os.path.exists(image_path):
        print(f"  WARNING: Image not found at {image_path}")
        return None, None, None

    cv_image = cv2.imread(image_path)
    if cv_image is None:
        print(f"  WARNING: Failed to read {image_path}")
        return None, None, None

    # Crop logic (same as simple_infer.py)
    diff = 1280 - 720
    left_crop = int(1 / 4 * diff)
    right_crop = int(3 / 4 * diff)

    h, w = cv_image.shape[:2]
    crop_info = {
        'left_crop': left_crop,
        'right_crop': right_crop,
        'original_h': h,
        'original_w': w
    }
    
    if w >= 1280 and h >= 720:
        cropped = cv_image[:, left_crop:1280 - right_crop]
    else:
        crop_w = min(720, w)
        crop_h = min(720, h)
        start_x = max(0, (w - crop_w) // 2)
        start_y = max(0, (h - crop_h) // 2)
        crop_info['center_crop_x'] = start_x
        crop_info['center_crop_y'] = start_y
        cropped = cv_image[start_y:start_y + crop_h, start_x:start_x + crop_w]

    cropped_rot = cv2.rotate(cropped, cv2.ROTATE_180)
    crop_info['crop_h'] = cropped_rot.shape[0]
    crop_info['crop_w'] = cropped_rot.shape[1]

    # Run inference
    preds = run_inference(model, cropped_rot)
    return preds, cv_image, crop_info, cropped_rot


def transform_landmarks_to_original(preds, crop_info):
    """Transform landmarks from cropped/rotated space to original image coordinates."""
    crop_h = crop_info['crop_h']
    crop_w = crop_info['crop_w']
    original_w = crop_info['original_w']
    original_h = crop_info['original_h']
    
    corrected = []
    for pred in preds:
        # Unpack x, y, and optionally visibility
        if len(pred) == 3:
            x, y, vis = pred
        else:
            x, y = pred
            vis = None
        
        # Reverse 180 degree rotation
        x_c = crop_w - x
        y_c = crop_h - y
        
        # Move slightly toward center (same as simple_infer.py)
        point = np.array([x_c, y_c])
        center = np.array([crop_w / 2.0, crop_h / 2.0])
        vec = (center - point)
        norm = np.linalg.norm(vec)
        if norm > 0:
            new_point = point + vec / norm * 60
        else:
            new_point = point
        
        # Add crop offset based on crop type
        if original_w >= 1280 and original_h >= 720:
            # Used left/right crop
            x_orig = new_point[0] + crop_info['left_crop']
        else:
            # Used center crop
            x_orig = new_point[0] + crop_info.get('center_crop_x', 0)
        
        y_orig = new_point[1] + crop_info.get('center_crop_y', 0)
        
        if vis is not None:
            corrected.append((float(x_orig), float(y_orig), vis))
        else:
            corrected.append((float(x_orig), float(y_orig)))
    
    return corrected


def annotate_image_with_landmarks(image: np.ndarray, landmarks, output_path: str):
    """Annotate image with landmark points and save it."""
    vis = image.copy()
    for idx, lm in enumerate(landmarks):
        # Handle both 2-value (x, y) and 3-value (x, y, visibility) tuples
        if len(lm) == 3:
            x, y, vis_score = lm
        else:
            x, y = lm
            vis_score = None
        
        xi, yi = int(round(x)), int(round(y))
        
        # Color based on visibility: green if positive, red if negative
        if vis_score is not None and vis_score < 0:
            color = (0, 0, 255)  # Red for low/negative visibility
        else:
            color = (0, 255, 0)  # Green for positive visibility
        
        cv2.circle(vis, (xi, yi), 5, color, -1)
        
        # Draw text with coordinates
        cv2.putText(vis, f"({xi},{yi})", (xi + 5, yi - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Draw landmark index
        cv2.putText(vis, f"{idx}", (xi - 10, yi + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, vis)


def main():
    print(f"Device: {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Input folder: {INPUT_FOLDER}\n")

    if not os.path.isdir(INPUT_FOLDER):
        print(f"ERROR: Input folder not found: {INPUT_FOLDER}")
        sys.exit(1)

    model = load_model(MODEL_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        f for f in Path(INPUT_FOLDER).rglob('*')
        if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"ERROR: No images found in {INPUT_FOLDER}")
        sys.exit(1)

    print(f"Found {len(image_files)} image(s). Processing...\n")

    # Storage for statistics
    all_visibility_scores = []  # All visibility scores from all landmarks in all images
    per_image_stats = []  # List of dicts with per-image statistics
    all_landmark_coords = []  # Collect all landmark coordinates across all images

    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing {image_path.name}...", end=" ")
        preds, cv_image, crop_info, cropped_rot = process_image(model, str(image_path))

        if preds is None:
            print("SKIPPED")
            continue

        # Extract visibility scores from this image
        vis_scores = [pred[2] for pred in preds]
        mean_vis = np.mean(vis_scores)
        var_vis = np.var(vis_scores)
        min_vis = np.min(vis_scores)
        max_vis = np.max(vis_scores)

        per_image_stats.append({
            'filename': image_path.name,
            'mean': mean_vis,
            'variance': var_vis,
            'min': min_vis,
            'max': max_vis,
            'scores': vis_scores
        })

        all_visibility_scores.extend(vis_scores)
        
        # Transform landmarks to original image coordinates for centroid analysis
        corrected_landmarks = transform_landmarks_to_original(preds, crop_info)
        
        # Collect landmark coordinates for centroid analysis
        for lm_idx, lm in enumerate(corrected_landmarks):
            x, y = lm[0], lm[1]
            all_landmark_coords.append({
                'lm_index': lm_idx,
                'x': x,
                'y': y,
                'image_idx': i - 1
            })
        
        # Annotate on the cropped/rotated image instead of original
        annotated_path = os.path.join(OUTPUT_DIR, 'annotated_images', f"{i:02d}_{image_path.stem}_annotated.png")
        annotate_image_with_landmarks(cropped_rot, preds, annotated_path)
        
        print(f"mean={mean_vis:.3f}, var={var_vis:.3f}")

    if not per_image_stats:
        print("ERROR: No images were successfully processed.")
        sys.exit(1)

    # Compute global statistics
    global_mean = np.mean(all_visibility_scores)
    global_var = np.var(all_visibility_scores)
    global_std = np.std(all_visibility_scores)

    # Calculate centroid and landmark distances from centroid per image
    landmark_coords_array = np.array([[lm['x'], lm['y']] for lm in all_landmark_coords])
    global_centroid = np.mean(landmark_coords_array, axis=0)
    
    # Calculate centroid and distances for each image
    image_centroid_stats = {}
    for img_idx in range(len(per_image_stats)):
        img_landmarks = [lm for lm in all_landmark_coords if lm['image_idx'] == img_idx]
        if img_landmarks:
            coords = np.array([[lm['x'], lm['y']] for lm in img_landmarks])
            img_centroid = np.mean(coords, axis=0)
            distances = np.linalg.norm(coords - img_centroid, axis=1)
            image_centroid_stats[img_idx] = {
                'centroid': img_centroid,
                'mean_distance': np.mean(distances),
                'std_distance': np.std(distances),
                'min_distance': np.min(distances),
                'max_distance': np.max(distances)
            }

    print(f"\n{'='*60}")
    print(f"GLOBAL STATISTICS (all {len(all_visibility_scores)} visibility scores):")
    print(f"  Mean: {global_mean:.4f}")
    print(f"  Variance: {global_var:.4f}")
    print(f"  Std Dev: {global_std:.4f}")
    print(f"  Min: {np.min(all_visibility_scores):.4f}")
    print(f"  Max: {np.max(all_visibility_scores):.4f}")
    print(f"{'='*60}")
    print(f"\nLANDMARK CENTROID ANALYSIS (per image):")
    print(f"  Global Centroid: ({global_centroid[0]:.1f}, {global_centroid[1]:.1f})")
    for img_idx, stats in image_centroid_stats.items():
        print(f"    Image {img_idx}: centroid=({stats['centroid'][0]:.0f}, {stats['centroid'][1]:.0f}), "
              f"mean_dist={stats['mean_distance']:.1f} px")
    print(f"{'='*60}\n")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Visibility Score Analysis", fontsize=16, fontweight='bold')

    # 1. Mean visibility per image
    ax = axes[0, 0]
    image_names = [s['filename'] for s in per_image_stats]
    means = [s['mean'] for s in per_image_stats]
    ax.bar(range(len(image_names)), means, color='steelblue', alpha=0.7)
    ax.axhline(y=global_mean, color='r', linestyle='--', linewidth=2, label=f'Global Mean: {global_mean:.3f}')
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Mean Visibility')
    ax.set_title('Mean Visibility Score per Image')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    if len(image_names) <= 20:
        ax.set_xticks(range(len(image_names)))
        ax.set_xticklabels(range(len(image_names)))

    # 2. Variance per image
    ax = axes[0, 1]
    variances = [s['variance'] for s in per_image_stats]
    ax.bar(range(len(image_names)), variances, color='coral', alpha=0.7)
    ax.axhline(y=global_var, color='r', linestyle='--', linewidth=2, label=f'Global Variance: {global_var:.3f}')
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Variance of Visibility')
    ax.set_title('Visibility Score Variance per Image')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    if len(image_names) <= 20:
        ax.set_xticks(range(len(image_names)))
        ax.set_xticklabels(range(len(image_names)))

    # 3. Histogram of all visibility scores
    ax = axes[1, 0]
    ax.hist(all_visibility_scores, bins=50, color='mediumseagreen', alpha=0.7, edgecolor='black')
    ax.axvline(x=global_mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {global_mean:.3f}')
    ax.axvline(x=0, color='k', linestyle='-', linewidth=1, alpha=0.5, label='0 (no visibility)')
    ax.set_xlabel('Visibility Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of All Visibility Scores')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 4. Box plot of visibility scores per image (if not too many images)
    ax = axes[1, 1]
    if len(per_image_stats) <= 15:
        vis_data = [s['scores'] for s in per_image_stats]
        bp = ax.boxplot(vis_data, labels=range(len(per_image_stats)), patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_xlabel('Image Index')
        ax.set_ylabel('Visibility Score')
        ax.set_title('Visibility Scores Distribution per Image')
        ax.grid(axis='y', alpha=0.3)
    else:
        # If too many images, show scatter plot instead
        x_coords = []
        y_coords = []
        for img_idx, stats in enumerate(per_image_stats):
            x_coords.extend([img_idx] * len(stats['scores']))
            y_coords.extend(stats['scores'])
        ax.scatter(x_coords, y_coords, alpha=0.5, s=20, color='purple')
        ax.set_xlabel('Image Index')
        ax.set_ylabel('Visibility Score')
        ax.set_title('Visibility Scores (Scatter Plot)')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'visibility_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")

    # Create second figure: scatter plot comparing visibility scores across images
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle("Visibility Score Comparison Across Images", fontsize=16, fontweight='bold')

    # Left: Scatter plot with image as x-axis, visibility score as y-axis
    ax = axes2[0]
    cmap = plt.cm.get_cmap('tab10' if len(per_image_stats) <= 10 else 'hsv')
    colors = [cmap(i / max(len(per_image_stats) - 1, 1)) for i in range(len(per_image_stats))]
    
    for img_idx, (stats, color) in enumerate(zip(per_image_stats, colors)):
        x_coords = [img_idx] * len(stats['scores'])
        y_coords = stats['scores']
        ax.scatter(x_coords, y_coords, alpha=0.6, s=80, color=color, 
                   label=f"Img {img_idx}: {stats['filename'][:15]}", edgecolors='black', linewidth=0.5)
    
    ax.axhline(y=global_mean, color='r', linestyle='--', linewidth=2, label=f'Global Mean: {global_mean:.3f}')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.3)
    ax.set_xlabel('Image Index', fontsize=11)
    ax.set_ylabel('Visibility Score', fontsize=11)
    ax.set_title('All Visibility Scores by Image')
    ax.grid(alpha=0.3)
    if len(per_image_stats) <= 15:
        ax.legend(fontsize=9, loc='best')

    # Right: Scatter plot with landmark index as x-axis, visibility score as y-axis (one series per image)
    ax = axes2[1]
    cmap = plt.cm.get_cmap('tab10' if len(per_image_stats) <= 10 else 'hsv')
    colors = [cmap(i / max(len(per_image_stats) - 1, 1)) for i in range(len(per_image_stats))]
    
    for img_idx, (stats, color) in enumerate(zip(per_image_stats, colors)):
        lm_indices = range(len(stats['scores']))
        ax.scatter(lm_indices, stats['scores'], alpha=0.6, s=80, color=color,
                   label=f"Img {img_idx}: {stats['filename'][:15]}", edgecolors='black', linewidth=0.5)
    
    ax.axhline(y=global_mean, color='r', linestyle='--', linewidth=2, label=f'Global Mean: {global_mean:.3f}')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.3)
    ax.set_xlabel('Landmark Index', fontsize=11)
    ax.set_ylabel('Visibility Score', fontsize=11)
    ax.set_title('Visibility Scores by Landmark (per image)')
    ax.grid(alpha=0.3)
    if len(per_image_stats) <= 15:
        ax.legend(fontsize=9, loc='best')

    plt.tight_layout()
    comparison_plot_path = os.path.join(OUTPUT_DIR, 'visibility_comparison_scatter.png')
    plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison scatter plot to {comparison_plot_path}")

    # Create third figure: landmark centroid distance analysis (per image)
    fig3, axes3 = plt.subplots(1, 2, figsize=(15, 5))
    fig3.suptitle("Landmark Distance from Centroid Analysis (Per Image)", fontsize=16, fontweight='bold')

    img_indices = sorted(image_centroid_stats.keys())
    mean_distances = [image_centroid_stats[i]['mean_distance'] for i in img_indices]
    std_distances = [image_centroid_stats[i]['std_distance'] for i in img_indices]
    image_labels = [f"Img {i}" for i in img_indices]

    # Left: Bar plot of mean distances from centroid per image
    ax = axes3[0]
    ax.bar(range(len(img_indices)), mean_distances, yerr=std_distances, capsize=5, color='darkgreen', alpha=0.7, error_kw={'linewidth': 2})
    overall_mean_dist = np.mean(mean_distances)
    ax.axhline(y=overall_mean_dist, color='r', linestyle='--', linewidth=2, label=f'Mean: {overall_mean_dist:.1f} px')
    ax.set_xlabel('Image Index', fontsize=11)
    ax.set_ylabel('Mean Distance from Centroid (px)', fontsize=11)
    ax.set_title('Mean Landmark Distance from Image Centroid')
    ax.set_xticks(range(len(img_indices)))
    ax.set_xticklabels(img_indices)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Right: Scatter plot showing all landmark positions with per-image centroids marked
    ax = axes3[1]
    cmap = plt.cm.get_cmap('tab10' if len(per_image_stats) <= 10 else 'hsv')
    colors = [cmap(i / max(len(per_image_stats) - 1, 1)) for i in range(len(per_image_stats))]
    
    for lm in all_landmark_coords:
        color = colors[lm['image_idx']]
        ax.scatter(lm['x'], lm['y'], alpha=0.4, s=30, color=color)
    
    # Plot per-image centroids
    for img_idx, stats in image_centroid_stats.items():
        color = colors[img_idx]
        ax.scatter(stats['centroid'][0], stats['centroid'][1], s=200, color=color, marker='*', 
                   edgecolors='black', linewidth=1.5, label=f'Img {img_idx}', zorder=5)
    
    ax.set_xlabel('X (pixels)', fontsize=11)
    ax.set_ylabel('Y (pixels)', fontsize=11)
    ax.set_title('Landmark Positions with Per-Image Centroids')
    if len(per_image_stats) <= 10:
        ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    centroid_plot_path = os.path.join(OUTPUT_DIR, 'landmark_centroid_analysis.png')
    plt.savefig(centroid_plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved centroid analysis plot to {centroid_plot_path}")

    # Save detailed results to CSV
    csv_path = os.path.join(OUTPUT_DIR, 'visibility_stats.csv')
    with open(csv_path, 'w') as f:
        f.write("Image,Mean_Visibility,Variance,Min,Max,Num_Landmarks\n")
        for stats in per_image_stats:
            f.write(f"{stats['filename']},{stats['mean']:.6f},{stats['variance']:.6f},"
                   f"{stats['min']:.6f},{stats['max']:.6f},{len(stats['scores'])}\n")
    print(f"Saved per-image stats to {csv_path}")

    # Save all individual visibility scores
    scores_path = os.path.join(OUTPUT_DIR, 'all_visibility_scores.txt')
    with open(scores_path, 'w') as f:
        f.write("Image_Index,Landmark_Index,Visibility_Score\n")
        for img_idx, stats in enumerate(per_image_stats):
            for lm_idx, vis in enumerate(stats['scores']):
                f.write(f"{img_idx},{lm_idx},{vis:.6f}\n")
    print(f"Saved all visibility scores to {scores_path}")

    print(f"\nAnalysis complete! Results saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
