#!/usr/bin/env python3
"""
Image Augmentation Script
Generates augmented copies of training images with random variations.

Usage:
    python augment_images.py --input ./data/images --output ./data/augmented_images --num-augmentations 5 --annos ./data/annos --output-annos ./data/augmented_annos
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import random
from tqdm import tqdm


def apply_random_augmentations(image, landmarks=None):
    """
    Apply random augmentations to an image and optionally transform landmarks.
    
    Args:
        image: PIL Image object
        landmarks: List of [x%, y%, visibility] landmarks (optional)
        
    Returns:
        Tuple of (augmented PIL Image, transformed landmarks or None)
    """
    img = image.copy()
    width, height = img.size
    
    # Track transformations that affect coordinates
    rotation_angle = 0
    horizontal_flip = False
    
    # Convert percentage landmarks to pixel coordinates for transformation
    pixel_landmarks = None
    if landmarks is not None:
        pixel_landmarks = []
        for lm in landmarks:
            x_pct, y_pct, vis = lm
            if vis > 0:  # Only process visible landmarks
                x_px = (x_pct / 100.0) * width
                y_px = (y_pct / 100.0) * height
                pixel_landmarks.append([x_px, y_px, vis])
            else:
                pixel_landmarks.append([0, 0, 0])  # Keep invisible landmarks as-is
    
    # Random padding
    if random.random() > 0.4:
        from PIL import ImageOps
        # Random padding on each side (0-15% of image dimension)
        max_pad_pct = 0.15
        pad_left = int(random.uniform(0, max_pad_pct) * width)
        pad_right = int(random.uniform(0, max_pad_pct) * width)
        pad_top = int(random.uniform(0, max_pad_pct) * height)
        pad_bottom = int(random.uniform(0, max_pad_pct) * height)
        
        # Random padding color (gray-ish)
        pad_color = (
            random.randint(100, 150),
            random.randint(100, 150),
            random.randint(100, 150)
        )
        
        # Apply padding
        img = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=pad_color)
        
        # Update landmarks to account for padding offset
        if pixel_landmarks:
            for lm in pixel_landmarks:
                if lm[2] > 0:  # Only adjust visible landmarks
                    lm[0] += pad_left
                    lm[1] += pad_top
        
        # Update dimensions
        width, height = img.size
    
    # Random rotation (small angles)
    if random.random() > 0.2:
        rotation_angle = random.uniform(-15, 15)
        img = img.rotate(rotation_angle, resample=Image.BICUBIC, expand=False, fillcolor=(128, 128, 128))
        
        if pixel_landmarks:
            # Rotate landmarks around image center
            import math
            center_x, center_y = width / 2, height / 2
            angle_rad = math.radians(rotation_angle)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            
            for lm in pixel_landmarks:
                if lm[2] > 0:  # Only rotate visible landmarks
                    # Translate to origin
                    x = lm[0] - center_x
                    y = lm[1] - center_y
                    
                    # Rotate (counter-clockwise for PIL rotation)
                    new_x = x * cos_a + y * sin_a
                    new_y = -x * sin_a + y * cos_a
                    
                    # Translate back
                    lm[0] = new_x + center_x
                    lm[1] = new_y + center_y
    
    # Random color adjustments (don't affect landmarks)
    if random.random() > 0.3:
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(factor)
    
    # Random brightness
    if random.random() > 0.3:
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(factor)
    
    # Random contrast
    if random.random() > 0.3:
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)
    
    # Random sharpness
    if random.random() > 0.5:
        factor = random.uniform(0.5, 2.0)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(factor)
    
    # Random blur
    if random.random() > 0.6:
        blur_radius = random.uniform(0.5, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Random noise (via jpeg compression artifacts)
    if random.random() > 0.5:
        import io
        quality = random.randint(70, 95)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img = Image.open(buffer)
    
    # Convert pixel landmarks back to percentages
    transformed_landmarks = None
    landmarks_outside = 0
    if pixel_landmarks:
        transformed_landmarks = []
        for idx, lm in enumerate(pixel_landmarks):
            if lm[2] > 0:
                # Check if landmark is outside image boundaries
                if lm[0] < 0 or lm[0] >= width or lm[1] < 0 or lm[1] >= height:
                    # Landmark is outside - set to invisible
                    transformed_landmarks.append([0, 0, 0])
                    landmarks_outside += 1
                else:
                    # Landmark is inside - keep it
                    x_pct = (lm[0] / width) * 100.0
                    y_pct = (lm[1] / height) * 100.0
                    transformed_landmarks.append([x_pct, y_pct, lm[2]])
            else:
                transformed_landmarks.append([0, 0, 0])
    
    return img, transformed_landmarks, landmarks_outside


def augment_images(input_dir, output_dir, num_augmentations, annos_dir=None, output_annos_dir=None, image_extensions=None):
    """
    Create augmented versions of all images in input directory.
    
    Args:
        input_dir: Path to directory containing original images
        output_dir: Path to directory where augmented images will be saved
        num_augmentations: Number of augmented versions to create per image
        annos_dir: Path to directory containing annotation JSON files (optional)
        output_annos_dir: Path to directory where augmented annotation files will be saved (optional)
        image_extensions: List of image file extensions to process
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create output annotations directory if needed
    annos_path = None
    output_annos_path = None
    use_annotations = False
    
    if annos_dir and output_annos_dir:
        annos_path = Path(annos_dir)
        output_annos_path = Path(output_annos_dir)
        output_annos_path.mkdir(parents=True, exist_ok=True)
        use_annotations = True
        print(f"📝 Will also create augmented annotation files")
    
    # Get all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    image_files = sorted(set(image_files))
    
    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return
    
    print(f"📁 Found {len(image_files)} images in {input_dir}")
    print(f"🎨 Creating {num_augmentations} augmented versions per image...")
    
    total_created = 0
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Load original image
            original_img = Image.open(img_path).convert('RGB')
            
            # Get filename components
            stem = img_path.stem
            ext = img_path.suffix
            
            # Load annotation if available
            anno_data = None
            if use_annotations and annos_path and output_annos_path:
                anno_file = annos_path / f"{stem}.json"
                if anno_file.exists():
                    with open(anno_file, 'r') as f:
                        anno_data = json.load(f)
            
            # Copy original to output directory
            original_output = output_path / f"{stem}{ext}"
            original_img.save(original_output)
            
            # Copy original annotation if available
            if anno_data and output_annos_path:
                original_anno_output = output_annos_path / f"{stem}.json"
                with open(original_anno_output, 'w') as f:
                    json.dump(anno_data, f, indent=2)
            
            # Create augmented versions
            for i in range(num_augmentations):
                # Apply augmentations with landmark transformation
                augmented_img, transformed_landmarks, landmarks_outside = apply_random_augmentations(
                    original_img, 
                    anno_data['landmarks'] if anno_data else None
                )
                
                # Print if landmarks went outside
                if landmarks_outside > 0:
                    print(f"  ⚠️  {stem}_aug{i+1:03d}: {landmarks_outside} landmark(s) moved outside image bounds")
                
                # Save with suffix
                aug_filename = f"{stem}_aug{i+1:03d}{ext}"
                aug_path = output_path / aug_filename
                augmented_img.save(aug_path)
                total_created += 1
                
                # Create corresponding annotation file
                if anno_data and output_annos_path and transformed_landmarks:
                    aug_anno_data = anno_data.copy()
                    aug_anno_data['image'] = aug_filename
                    aug_anno_data['landmarks'] = transformed_landmarks
                    
                    aug_anno_filename = f"{stem}_aug{i+1:03d}.json"
                    aug_anno_path = output_annos_path / aug_anno_filename
                    
                    with open(aug_anno_path, 'w') as f:
                        json.dump(aug_anno_data, f, indent=2)
                
        except Exception as e:
            print(f"\n⚠️  Error processing {img_path.name}: {e}")
            continue
    
    print(f"\n✅ Done! Created {total_created} augmented images")
    print(f"📂 Output directory: {output_dir}")
    print(f"📊 Total images (original + augmented): {len(image_files) + total_created}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate augmented versions of training images with random variations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input directory containing original images'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for augmented images'
    )
    
    parser.add_argument(
        '--num-augmentations', '-n',
        type=int,
        default=5,
        help='Number of augmented versions to create per image'
    )
    
    parser.add_argument(
        '--annos', '-a',
        type=str,
        default=None,
        help='Input directory containing annotation JSON files (optional)'
    )
    
    parser.add_argument(
        '--output-annos',
        type=str,
        default=None,
        help='Output directory for augmented annotation files (optional, requires --annos)'
    )
    
    parser.add_argument(
        '--extensions', '-e',
        type=str,
        nargs='+',
        default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'],
        help='Image file extensions to process'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (optional)'
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"🎲 Using random seed: {args.seed}")
    
    # Run augmentation
    augment_images(
        input_dir=args.input,
        output_dir=args.output,
        num_augmentations=args.num_augmentations,
        annos_dir=args.annos,
        output_annos_dir=args.output_annos,
        image_extensions=args.extensions
    )


if __name__ == '__main__':
    main()
