import cv2
import numpy as np
import matplotlib.pyplot as plt

def fit_color_model(image_path, method="mahalanobis", threshold_percentile=90, visualize=True):
    """
    Fit a color model based on non-black pixels in an image.
    Returns mean, covariance, and automatic distance threshold.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Step 1: Isolate non-black pixels ---
    mask_nonblack = np.linalg.norm(img_rgb, axis=2) > 30
    if np.count_nonzero(mask_nonblack) == 0:
        raise ValueError("No non-black pixels detected — check your image or threshold.")

    pixels = img_rgb[mask_nonblack].astype(np.float32)

    # --- Step 2: Compute mean and covariance ---
    mean = np.mean(pixels, axis=0)
    cov = np.cov(pixels, rowvar=False) + np.eye(3) * 1e-5
    inv_cov = np.linalg.inv(cov)

    # --- Step 3: Compute distances for non-black pixels ---
    diffs = pixels - mean
    if method == "mahalanobis":
        dists = np.sqrt(np.sum((diffs @ inv_cov) * diffs, axis=1))
    else:
        dists = np.linalg.norm(diffs, axis=1)

    dists_norm = (dists - dists.min()) / (dists.max() - dists.min() + 1e-6)
    threshold = np.percentile(dists_norm, threshold_percentile)

    if visualize:
        plt.figure(figsize=(6,4))
        plt.hist(dists_norm, bins=50, color='gray')
        plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold = {threshold:.3f}')
        plt.title(f"{method.capitalize()} distance distribution")
        plt.xlabel("Normalized distance")
        plt.ylabel("Count")
        plt.legend()
        plt.show()

    print(f"Model fitted: mean={mean.round(2)}, threshold={threshold:.3f}")
    return mean, cov, threshold


def apply_color_model(image_rgb, mean, cov, threshold, method="mahalanobis", visualize=True):
    """
    Apply a learned color model to a new image.
    """
    img_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

    # Compute distances for all pixels
    inv_cov = np.linalg.inv(cov)
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    diffs = pixels - mean

    if method == "mahalanobis":
        dists = np.sqrt(np.sum((diffs @ inv_cov) * diffs, axis=1))
    else:
        dists = np.linalg.norm(diffs, axis=1)

    dists_norm = (dists - dists.min()) / (dists.max() - dists.min() + 1e-6)
    mask = (dists_norm < threshold).astype(np.uint8).reshape(img_rgb.shape[:2]) * 255

    if visualize:
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.imshow(img_rgb)
        plt.title("New Image")
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(dists_norm.reshape(img_rgb.shape[:2]), cmap='inferno')
        plt.title(f"{method.capitalize()} Distance Map")
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.imshow(mask, cmap='gray')
        plt.title("Resulting Mask")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return mask
