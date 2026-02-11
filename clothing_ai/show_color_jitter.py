import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import random
import glob
from typing import List, Optional

# ====================================================================
# A) CONFIGURATION
# ====================================================================

# Specify image path(s) here, or leave as None to use random images
SINGLE_IMAGE_PATH: Optional[List[str]] = [
    "clothing_ai/data/our_flats/LsoUeFc-.jpeg"
    # "clothing_ai/data/deepFashion2/validation/images/032098.jpg",
]

# Used if SINGLE_IMAGE_PATH is None
DATA_DIR: str = "clothing_ai/data/deepFashion2/validation"

# Only used if SINGLE_IMAGE_PATH is None
NUM_IMAGES: int = 2 # Number of base images to process

# --- VISUALIZATION CONFIG ---
OUTPUT_DIR: str = "color_jitter_visualizations_only_variants" # Changed output folder name
NUM_JITTERED_VARIANTS: int = 3 # Increased to 6 variants for better comparison
IMG_SIZE: int = 224 # Target size for visualization

# --- COLOR JITTER PARAMETERS ---
JITTER_PARAMS = (0.5, 0.5, 0.5, 0.1)

# ====================================================================
# B) SETUP & IMAGE SELECTION
# ====================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output folder created: {OUTPUT_DIR}")

# 1. Color Jitter Transform
color_jitter_transform = transforms.ColorJitter(*JITTER_PARAMS)

# 2. General Transform for initial processing
base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
])

# 3. Image Selection Logic (kept the same)
def get_selected_images(single_path: Optional[List[str]], data_dir: str, num_images: int) -> List[str]:
    if single_path is not None:
        if isinstance(single_path, str):
            return [single_path]
        return single_path
    
    image_files = []
    patterns = [
        os.path.join(data_dir, "images", "*.jpg"),
        os.path.join(data_dir, "images", "*.png"),
        os.path.join(data_dir, "*.jpg"),
        os.path.join(data_dir, "*.png"),
    ]
    
    for pattern in patterns:
        found = glob.glob(pattern)
        if found:
            image_files.extend(found)
            
    if not image_files:
        raise FileNotFoundError(f"No images found in {data_dir}.")
    
    return random.sample(image_files, min(num_images, len(image_files)))

try:
    selected_images = get_selected_images(SINGLE_IMAGE_PATH, DATA_DIR, NUM_IMAGES)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure your image paths (clothing_ai/data/...) are correct or modify DATA_DIR.")
    exit()

print(f"\nProcessing {len(selected_images)} base image(s).")
print(f"Applying Color Jitter with parameters: {JITTER_PARAMS}")

# ====================================================================
# C) VISUALIZATION FUNCTION (MODIFIED)
# ====================================================================

def visualize_color_jitter(img_path: str, base_transform, jitter_transform, num_variants: int, output_dir: str):
    """Loads an image, applies color jitter repeatedly, creates a visualization 
       grid of *only* the jitted variants, displays it, and saves it."""
       
    image_id = os.path.splitext(os.path.basename(img_path))[0]
    
    try:
        # Load and resize the base image (PIL format)
        orig_pil = Image.open(img_path).convert("RGB")
        resized_pil = base_transform(orig_pil)
    except FileNotFoundError:
        print(f"Skipping image: {img_path} not found.")
        return
    except Exception as e:
        print(f"Skipping image: Error loading/processing {img_path}: {e}")
        return

    # Prepare the list of images for the grid: ONLY JITTERED VARIANTS
    image_list = []
    
    # Generate the jitted variants
    for _ in range(num_variants):
        # Apply the random jitter transformation
        jitted_pil = jitter_transform(resized_pil)
        image_list.append(jitted_pil)

    # Grid dimensions: 1 row x num_variants columns
    cols = num_variants
    rows = 1
    
    # Create the visualization canvas
    grid_width = IMG_SIZE * cols
    grid_height = IMG_SIZE * rows
    
    grid_canvas_pil = Image.new('RGB', (grid_width, grid_height))

    for idx, pil_img in enumerate(image_list):
        col = idx % cols
        x_start = col * IMG_SIZE
        
        # Paste the image onto the canvas
        grid_canvas_pil.paste(pil_img, (x_start, 0))

    # Convert PIL Image to OpenCV format (BGR) for display/saving
    grid_canvas_cv = cv2.cvtColor(np.array(grid_canvas_pil), cv2.COLOR_RGB2BGR)

    # --- Display ---
    window_name = f"Color Jitter VARIANTS ONLY: {image_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, grid_canvas_cv)
    
    # --- Save ---
    output_filename = os.path.join(output_dir, f"jitter_variants_{image_id}.jpg")
    cv2.imwrite(output_filename, grid_canvas_cv)
    print(f"Saved visualization to: {output_filename}")


# ====================================================================
# D) EXECUTION
# ====================================================================

for img_path in selected_images:
    visualize_color_jitter(
        img_path, 
        base_transform, 
        color_jitter_transform, 
        NUM_JITTERED_VARIANTS, 
        OUTPUT_DIR
    )

print("\n--- Display Complete ---")
print(f"Showing all windows. Press any key on an image window to close all.")

# Hold the windows open until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()