
import os
import cv2
from ament_index_python.packages import get_package_share_directory
import numpy as np

def segment_foreground(image, background):
    """
    Segment the foreground of an image
    """
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv_image)

    white_mask = cv2.inRange(background, (255, 255, 255), (255, 255, 255))
    s[white_mask == 255] = 0
    
    blur = cv2.GaussianBlur(s, (11,11), 0)

    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh = cv2.bitwise_not(thresh)

    kernel = np.ones((11,11), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    clean = cv2.bitwise_not(clean)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    central_contour = None
    
    if contours:
        h_img, w_img = image.shape[:2]
        image_center = np.array([w_img / 2, h_img / 2])

        min_dist = float('inf')
        central_contour = None

        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroid = np.array([cx, cy])
                dist = np.linalg.norm(centroid - image_center)
                if dist < min_dist:
                    min_dist = dist
                    central_contour = cnt

    forergound_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(forergound_mask, [central_contour], -1, 255, -1)
    
    return forergound_mask, central_contour


def load_background_image(image_name='background.png'):
    """
    Load the background image
    """
    
    pkg_path = get_package_share_directory('image_processing_pkg')
    image_path = os.path.join(pkg_path, 'data', image_name)
    
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    return img


def step_1_instructions(cv_image, background):
    forergound_mask, central_contour = segment_foreground(cv_image, background)
    
    hull_points = cv2.convexHull(central_contour)  # Nx1x2
    hull_points = hull_points.reshape(-1, 2)

    # --- Approximate hull to hexagon ---
    # Use iterative approxPolyDP to reduce to 6 points
    epsilon = 0.01 * cv2.arcLength(hull_points, True)
    for i in range(100):
        approx = cv2.approxPolyDP(hull_points, epsilon, True)
        if len(approx) <= 4:
            break
        epsilon *= 1.05

    # If we got fewer than 6 points, resample evenly
    if len(approx) < 4:
        approx = approx[np.linspace(0, len(approx)-1, 6, dtype=int)]

    hexagon = approx.reshape(-1, 2)

    # Draw hexagon on image
    image_with_hull = cv_image.copy()
    distances = []
    for i in range(4):
        pt1 = tuple(hexagon[i])
        pt2 = tuple(hexagon[(i + 1) % 4])
        distances.append(float(np.linalg.norm(np.array(pt1) - np.array(pt2))))
        cv2.line(image_with_hull, pt1, pt2, (0, 0, 255), 2)

    point_occurences = {}

    indices = np.argsort(np.array(distances))[-3:]
    for idx in indices:
        pt1 = tuple(hexagon[idx])
        pt2 = tuple(hexagon[(idx + 1) % 4])
        if pt1 in point_occurences:
            point_occurences[pt1] += 1
        else:
            point_occurences[pt1] = 1
        
        if pt2 in point_occurences:
            point_occurences[pt2] += 1
        else:
            point_occurences[pt2] = 1
            
        cv2.line(image_with_hull, pt1, pt2, (0, 255, 0), 2)
        
    shirt_corners = [np.array(pt) for pt, count in point_occurences.items() if count > 1]

    M = cv2.moments(central_contour)

    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0  # Avoid division by zero for degenerate contours

    cv2.circle(image_with_hull, (cx, cy), 5, (255, 0, 0), -1)

    pick_point = [0, 0]
    place_point = [0, 0]
    
    if len(shirt_corners) >= 2:
        mid_vec = np.array([cx, cy]) - shirt_corners[0]
        pick_point= (np.round(shirt_corners[0] + 20 * (mid_vec / np.linalg.norm(mid_vec))).astype(int)).tolist()
        
        direction_vec = shirt_corners[1] - shirt_corners[0]
        place_point = (np.round(shirt_corners[0] + 2/3 * direction_vec).astype(int)).tolist()
        cv2.circle(image_with_hull, pick_point, 5, (255, 0, 0), -1)
        cv2.circle(image_with_hull, place_point, 5, (0, 0, 255), -1)
        
    #cv2.imwrite('/home/peter/uni/project_clothing_fresh/image_analysis/images/image_1_instructions.png', image_with_hull)
    #print(pick_point, place_point)
    
    return pick_point, place_point
    

def step_2_instructions(cv_image, background):
    forergound_mask, central_contour = segment_foreground(cv_image, background)
    
    pick_point = [0, 0]
    place_point = [0, 0]
    return pick_point, place_point

def step_3_instructions(cv_image, background):
    forergound_mask, central_contour = segment_foreground(cv_image, background)
    
    pick_point = [0, 0]
    place_point = [0, 0]
    return pick_point, place_point

def step_4_instructions(cv_image, background):
    forergound_mask, central_contour = segment_foreground(cv_image, background)
    
    pick_point = [0, 0]
    place_point = [0, 0]
    return pick_point, place_point

def step_5_instructions(cv_image, background):
    forergound_mask, central_contour = segment_foreground(cv_image, background)
    
    pick_point = [0, 0]
    place_point = [0, 0]
    return pick_point, place_point

def step_6_instructions(cv_image, background):
    forergound_mask, central_contour = segment_foreground(cv_image, background)
    
    pick_point = [0, 0]
    place_point = [0, 0]
    return pick_point, place_point


def main():
    cv_image = cv2.imread("/home/peter/uni/project_clothing_fresh/image_analysis/images/1.png")
    back_ground = cv2.imread("/home/peter/uni/project_clothing_fresh/image_analysis/images/background.png")
    pick_point, place_point = step_1_instructions(cv_image, back_ground)
    
if __name__ == "__main__":
    main()