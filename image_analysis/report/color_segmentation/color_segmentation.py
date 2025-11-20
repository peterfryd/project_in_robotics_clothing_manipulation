import cv2
import numpy as np

def build_color_model_from_mask(mask_image_path):
    """
    mask_image_path: image where non-white pixels contain shirt colors
    """
    mask_img = cv2.imread(mask_image_path)
    if mask_img is None:
        raise ValueError("Could not load mask image.")

    # Convert mask to grayscale to find non-white pixels
    gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    non_white_idx = np.where(gray < 250)  # non-white pixels

    # Extract HSV colors from non-white pixels
    hsv_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2HSV)
    shirt_pixels = hsv_mask[non_white_idx]

    mean_color = np.mean(shirt_pixels, axis=0)
    cov_color = np.cov(shirt_pixels.T)

    # HSV bounds for inRange
    h_mean, s_mean, v_mean = mean_color
    h_std, s_std, v_std = np.std(shirt_pixels, axis=0)
    lower = np.array([max(0, h_mean-2*h_std), max(0, s_mean-2*s_std), max(0, v_mean-2*v_std)], dtype=np.uint8)
    upper = np.array([min(179, h_mean+2*h_std), min(255, s_mean+2*s_std), min(255, v_mean+2*v_std)], dtype=np.uint8)

    return lower, upper, mean_color, cov_color

def segment_current_image(current_image_path, lower, upper, mean_color, cov_color, extra_mask_path=None):
    """
    Segment shirt in a new image using three methods:
    - inRange (HSV bounds)
    - Euclidean distance
    - Mahalanobis distance
    Apply extra_mask if provided (black = remove)
    """
    current = cv2.imread(current_image_path)
    if current is None:
        raise ValueError("Could not load current image.")

    hsv_current = cv2.cvtColor(current, cv2.COLOR_BGR2HSV)

    # --- Method 1: inRange ---
    mask_inrange = cv2.inRange(hsv_current, lower, upper)

    # --- Method 2: Euclidean distance ---
    diff = hsv_current.astype(np.float32) - mean_color
    dist_euc = np.linalg.norm(diff, axis=2)
    mask_euc = np.zeros_like(dist_euc, dtype=np.uint8)
    mask_euc[dist_euc < 60] = 255

    # --- Method 3: Mahalanobis distance ---
    cov_inv = np.linalg.inv(cov_color)
    mask_maha = np.zeros(current.shape[:2], dtype=np.uint8)
    for i in range(current.shape[0]):
        for j in range(current.shape[1]):
            x = hsv_current[i,j,:].astype(np.float32)
            delta = x - mean_color
            d = np.sqrt(delta @ cov_inv @ delta.T)
            if d < 3:
                mask_maha[i,j] = 255

    # --- Apply extra mask if provided ---
    if extra_mask_path is not None:
        extra_mask_img = cv2.imread(extra_mask_path, cv2.IMREAD_GRAYSCALE)
        if extra_mask_img is None:
            raise ValueError("Could not load extra mask image.")
        # Create binary mask: 0 = remove, non-zero = keep
        _, extra_bin = cv2.threshold(extra_mask_img, 1, 255, cv2.THRESH_BINARY)
        # Apply bitwise_and
        mask_inrange = cv2.bitwise_and(mask_inrange, extra_bin)
        mask_euc = cv2.bitwise_and(mask_euc, extra_bin)
        mask_maha = cv2.bitwise_and(mask_maha, extra_bin)

    # Apply masks to original image
    result_inrange = cv2.bitwise_and(current, current, mask=mask_inrange)
    result_euc = cv2.bitwise_and(current, current, mask=mask_euc)
    result_maha = cv2.bitwise_and(current, current, mask=mask_maha)

    # Show results
    #cv2.imshow("Current Image", current)
    #cv2.imshow("Mask inRange", mask_inrange)
    cv2.imshow("Segmented inRange", result_inrange)
    #cv2.imshow("Mask Euclidean", mask_euc)
    cv2.imshow("Segmented Euclidean", result_euc)
    #cv2.imshow("Mask Mahalanobis", mask_maha)
    cv2.imshow("Segmented Mahalanobis", result_maha)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("segmented_inrange.png", result_inrange)
    cv2.imwrite("segmented_euclidean.png", result_euc)
    cv2.imwrite("segmented_mahalanobis.png", result_maha)
    
    
if __name__ == "__main__":
    mask_image = "anno.png"         # annotated shirt mask
    current_image = "current.png"   # new image
    extra_mask = "mask.png"   # black pixels = remove

    lower, upper, mean_color, cov_color = build_color_model_from_mask(mask_image)
    segment_current_image(current_image, lower, upper, mean_color, cov_color, extra_mask_path=extra_mask)
