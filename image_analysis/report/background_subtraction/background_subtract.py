import cv2
import numpy as np

def background_subtraction(background_path, image_path, extra_mask_path=None, threshold=40):
    # Load images
    bg = cv2.imread(background_path)
    img = cv2.imread(image_path)

    if bg is None or img is None:
        raise ValueError("Could not load images. Check file paths.")

    # Ensure same dimensions
    if bg.shape != img.shape:
        raise ValueError("Background and input image must have the same size.")

    # Convert to grayscale
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    # Absolute difference
    diff = cv2.absdiff(bg_gray, img_gray)

    # Threshold to get foreground mask
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_DILATE, kernel)

    # Apply extra mask if provided
    if extra_mask_path is not None:
        extra_mask = cv2.imread(extra_mask_path, cv2.IMREAD_GRAYSCALE)
        if extra_mask is None:
            raise ValueError("Could not load extra mask image.")
        if extra_mask.shape != mask_clean.shape:
            raise ValueError("Extra mask must have the same dimensions as images.")
        # Combine masks (logical AND)
        mask_clean = cv2.bitwise_and(mask_clean, extra_mask)

    # Apply mask to original image
    foreground = cv2.bitwise_and(img, img, mask=mask_clean)

    img_test = cv2.bitwise_and(img, img, mask=extra_mask)
    cv2.imwrite("test_extra_mask_application.png", img_test)
    # Show results
    cv2.imshow("Background", bg)
    cv2.imshow("Current Image", img)
    cv2.imshow("Difference", diff)
    cv2.imshow("Foreground Mask", mask_clean)
    cv2.imshow("Foreground Extracted", foreground)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("foreground_mask.png", mask_clean)
    cv2.imwrite("difference.png", diff)
    
    return mask_clean, foreground


if __name__ == "__main__":
    background_path = "background.png"
    image_path = "current.png"
    extra_mask_path = "mask.png"  # Optional additional mask

    mask, fg = background_subtraction(background_path, image_path, extra_mask_path, threshold=40)
