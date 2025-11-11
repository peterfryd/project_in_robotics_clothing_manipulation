#!/usr/bin/env python3
"""
Annotation Visualization Script
Displays random images with their landmark annotations plotted on them.

Usage:
    python visualize_annotations.py --images ./data/images --annos ./data/annos --num-samples 5
    python visualize_annotations.py --images ./data/augmented_images --annos ./data/augmented_annos --num-samples 10
"""

import os
import json
import argparse
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def load_annotation(anno_path):
    """Load annotation data from JSON file."""
    with open(anno_path, 'r') as f:
        return json.load(f)


def plot_landmarks_on_image(img_path, anno_data, ax=None):
    """
    Plot image with landmarks overlaid.
    
    Args:
        img_path: Path to image file
        anno_data: Dictionary containing annotation data
        ax: Matplotlib axis to plot on (optional)
    
    Returns:
        Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Load and display image
    img = Image.open(img_path)
    ax.imshow(img)
    
    width, height = img.size
    landmarks = anno_data.get('landmarks', [])
    
    # Plot landmarks
    for idx, landmark in enumerate(landmarks):
        x_pct, y_pct, visibility = landmark
        
        if visibility > 0:  # Only plot visible landmarks
            # Convert percentage to pixel coordinates
            x_px = (x_pct / 100.0) * width
            y_px = (y_pct / 100.0) * height
            
            # Plot landmark point
            ax.plot(x_px, y_px, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=1.5)
            
            # Add landmark number (1-indexed)
            ax.text(x_px, y_px - 10, str(idx + 1), 
                   color='yellow', fontsize=10, fontweight='bold',
                   ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    # Add title with metadata
    category = anno_data.get('category_name', 'Unknown')
    num_landmarks = anno_data.get('num_landmarks', len(landmarks))
    num_visible = sum(1 for lm in landmarks if lm[2] > 0)
    
    title = f"{anno_data['image']}\n{category} | {num_visible}/{num_landmarks} landmarks visible"
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    
    return ax


def visualize_random_samples(images_dir, annos_dir, num_samples=5, seed=None):
    """
    Display random samples with annotations.
    
    Args:
        images_dir: Directory containing images
        annos_dir: Directory containing annotation JSON files
        num_samples: Number of random samples to display
        seed: Random seed for reproducibility (optional)
    """
    images_path = Path(images_dir)
    annos_path = Path(annos_dir)
    
    # Get all annotation files
    anno_files = sorted(annos_path.glob('*.json'))
    
    if not anno_files:
        print(f"❌ No annotation files found in {annos_dir}")
        return
    
    print(f"📁 Found {len(anno_files)} annotation files")
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        print(f"🎲 Using random seed: {seed}")
    
    # Sample random annotations
    num_samples = min(num_samples, len(anno_files))
    selected_annos = random.sample(anno_files, num_samples)
    
    print(f"🎨 Visualizing {num_samples} random samples...")
    
    # Calculate grid layout
    cols = min(3, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    
    # Handle single subplot case
    if num_samples == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot each sample
    for idx, anno_file in enumerate(selected_annos):
        try:
            # Load annotation
            anno_data = load_annotation(anno_file)
            
            # Find corresponding image
            img_filename = anno_data['image']
            img_path = images_path / img_filename
            
            if not img_path.exists():
                print(f"⚠️  Image not found: {img_path}")
                axes[idx].text(0.5, 0.5, f"Image not found:\n{img_filename}",
                             ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].axis('off')
                continue
            
            # Plot image with landmarks
            plot_landmarks_on_image(img_path, anno_data, axes[idx])
            
        except Exception as e:
            print(f"⚠️  Error processing {anno_file.name}: {e}")
            axes[idx].text(0.5, 0.5, f"Error:\n{str(e)}",
                         ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Visualization complete!")


def visualize_specific_image(image_path, anno_path):
    """
    Visualize a specific image with its annotations.
    
    Args:
        image_path: Path to image file
        anno_path: Path to annotation JSON file
    """
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    if not Path(anno_path).exists():
        print(f"❌ Annotation not found: {anno_path}")
        return
    
    # Load annotation
    anno_data = load_annotation(anno_path)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Plot image with landmarks
    plot_landmarks_on_image(image_path, anno_data, ax)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Visualization complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize images with landmark annotations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--images', '-i',
        type=str,
        required=True,
        help='Directory containing images'
    )
    
    parser.add_argument(
        '--annos', '-a',
        type=str,
        required=True,
        help='Directory containing annotation JSON files'
    )
    
    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=5,
        help='Number of random samples to display'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (optional)'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Specific image file to visualize (optional, overrides random sampling)'
    )
    
    parser.add_argument(
        '--anno',
        type=str,
        default=None,
        help='Specific annotation file for the image (required if --image is used)'
    )
    
    args = parser.parse_args()
    
    # Check if specific image visualization is requested
    if args.image:
        if not args.anno:
            print("❌ Error: --anno is required when using --image")
            return
        visualize_specific_image(args.image, args.anno)
    else:
        # Random sampling
        visualize_random_samples(
            images_dir=args.images,
            annos_dir=args.annos,
            num_samples=args.num_samples,
            seed=args.seed
        )


if __name__ == '__main__':
    main()
