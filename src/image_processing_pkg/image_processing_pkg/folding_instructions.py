
from cmath import rect
from math import inf
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

    foregroundd_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(foregroundd_mask, [central_contour], -1, 255, -1)
    
    return foregroundd_mask, central_contour


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


def step_1_instructions(landmarks:np.ndarray,  fold_type:str='square') -> tuple[list, list]:
    if fold_type == 'star':
        # Pick point is landmark 3
        # Place point is midpoint between landmark 1, 6 and 8
        pick_point = [landmarks[3].x, landmarks[3].y]
        place_point_y = (landmarks[1].y + landmarks[6].y + landmarks[8].y) / 3
        place_point_x = (landmarks[1].x + landmarks[6].x + landmarks[8].x) / 3
        place_point = [place_point_x, place_point_y]
    elif fold_type == 'square':
        # Pick point is landmark 3
        # Place point is two thirds between landmark 3 and 6
        pick_point = [landmarks[3].x, landmarks[3].y]
        place_point_x = (landmarks[6].x - landmarks[3].x)*2/3 + landmarks[3].x
        place_point_y = (landmarks[6].y - landmarks[3].y)*2/3 + landmarks[3].y
        place_point = [place_point_x, place_point_y]
        pick_point = [landmarks[0].x, landmarks[0].y]
        place_point = [landmarks[1].x, landmarks[1].y]
    else:
        raise ValueError(f"Unknown fold type: {fold_type}")

    return pick_point, place_point
    

def step_2_instructions(landmarks:np.ndarray, landmarks_origional:np.ndarray, fold_type:str='square') -> tuple[np.ndarray, np.ndarray]:
    if fold_type == 'star':
        # Pick point is landmark 6
        # Place point is midpoint between landmark 6 and landmarks_origional 1
        pick_point = landmarks[6]
        place_point = (landmarks[6] - landmarks_origional[1])*0.5 + landmarks_origional[1]
    elif fold_type == 'square':
        # Pick point is landmark 1
        # Place point is two thirds from landmark_origional 1 to 8
        pick_point = landmarks[1]
        place_point = (landmarks_origional[8] - landmarks_origional[1])*2/3 + landmarks_origional[1]
    else:
        raise ValueError(f"Unknown fold type: {fold_type}")

    return pick_point, place_point


def step_3_instructions(landmarks:np.ndarray, landmarks_origional:np.ndarray, fold_type:str='square') -> tuple[np.ndarray, np.ndarray]:
    if fold_type == 'star':
        # Pick point is landmark 1
        # Place point is the midpoint between landmark 1 and landmarks_origional 6
        pick_point = landmarks[1]
        place_point = (landmarks_origional[6] - landmarks[1])*0.5 + landmarks[1]
    elif fold_type == 'square':
        # Pick point is landmark 6
        # Place point is landmarks_origional 4
        pick_point = landmarks[6]
        place_point = landmarks_origional[4]
    else:
        raise ValueError(f"Unknown fold type: {fold_type}")
    
    return pick_point, place_point


def step_4_instructions(landmarks:np.ndarray, landmarks_origional:np.ndarray, fold_type:str='square') -> tuple[np.ndarray, np.ndarray]:
    if fold_type == 'star':
        # Pick point is landmark 8
        # Place point is midpoint between landmark 8 and 3
        pick_point = landmarks[8]
        place_point = (landmarks_origional[8] - landmarks_origional[3])*0.5 + landmarks_origional[3]
    elif fold_type == 'square':
        # Pick point is landmark 8
        # Place point is point two thirds from landmarks_origional 8 to 1
        pick_point = landmarks[8]
        place_point = (landmarks_origional[1] - landmarks_origional[8])*2/3 + landmarks_origional[8]
    else:
        raise ValueError(f"Unknown fold type: {fold_type}")
    
    return pick_point, place_point


def step_5_instructions(landmarks:np.ndarray, landmarks_origional:np.ndarray, fold_type:str='square') -> tuple[np.ndarray, np.ndarray] | None:
    if fold_type == 'star':
        print("Star fold only has 4 steps")
        return None
    elif fold_type == 'square':
        # Pick point is between landmarks_origional 1 and 8
        # Place point is between landmarks_origional 4 and 5
        pick_point = (landmarks_origional[1] - landmarks_origional[8])*0.5 + landmarks_origional[8]
        place_point = (landmarks_origional[4] - landmarks_origional[5])*0.5 + landmarks_origional[5]
    else:
        raise ValueError(f"Unknown fold type: {fold_type}")
    
    return pick_point, place_point


def main():
    FOLD_TYPE = 'square'  # 'star' or 'square'



    if FOLD_TYPE not in ['star', 'square']:
        raise ValueError(f"Unknown fold type: {FOLD_TYPE}")
    images_path = f"saved_images/{FOLD_TYPE}_fold"
    cv_image_1 = cv2.imread(images_path + "/image_1.png")
    cv_image_2 = cv2.imread(images_path + "/image_2.png")
    cv_image_3 = cv2.imread(images_path + "/image_3.png")
    cv_image_4 = cv2.imread(images_path + "/image_4.png")
    cv_image_5 = cv2.imread(images_path + "/image_5.png")
    cv_image_6 = cv2.imread(images_path + "/image_6.png")
    

    # Square fold landmarks
    if FOLD_TYPE == 'square':
        landmarks = np.array([
            [0,0],
            [880, 550],
            [440, 650],
            [310, 710],
            [200, 450],
            [200, 300],
            [275, 10],
            [420, 50],
            [840, 50]
        ])
    # Star fold landmarks
    else:
        landmarks = np.array([
                  [0,0],
            [810, 600],
            [380, 650],
            [250, 710],
            [150, 430],
            [160, 280],
            [220, 40],
            [380, 60],
            [880, 100]
        ])
    
    landmarks_origional = landmarks.copy()

    for point in landmarks:
        for image in [cv_image_1, cv_image_2, cv_image_3, cv_image_4, cv_image_5]:
            cv2.circle(image, tuple(point.astype(int)), 5, (100, 100, 100), -1)

    # get pick points for step 1
    pick_point_1, place_point_1 = step_1_instructions(landmarks, fold_type=FOLD_TYPE)
    # Plot pickpoint on image
    cv2.circle(cv_image_1, tuple(pick_point_1.astype(int)), 10, (0, 255, 0), -1)
    cv2.circle(cv_image_1, tuple(place_point_1.astype(int)), 10, (255, 0, 0), -1)
    cv2.imshow("Image 1 with pick point", cv_image_1)
    cv2.waitKey(0)

    # get pick points for step 2
    pick_point_2, place_point_2 = step_2_instructions(landmarks, landmarks_origional, fold_type=FOLD_TYPE)
    # Plot pickpoint on image
    cv2.circle(cv_image_2, tuple(pick_point_2.astype(int)), 10, (0, 255, 0), -1)
    cv2.circle(cv_image_2, tuple(place_point_2.astype(int)), 10, (255, 0, 0), -1)
    cv2.imshow("Image 2 with pick point", cv_image_2)
    cv2.waitKey(0)

    # get pick points for step 3
    pick_point_3, place_point_3 = step_3_instructions(landmarks, landmarks_origional, fold_type=FOLD_TYPE)
    # Plot pickpoint on image
    cv2.circle(cv_image_3, tuple(pick_point_3.astype(int)), 10, (0, 255, 0), -1)
    cv2.circle(cv_image_3, tuple(place_point_3.astype(int)), 10, (255, 0, 0), -1)
    cv2.imshow("Image 3 with pick point", cv_image_3)
    cv2.waitKey(0)

    # get pick points for step 4
    pick_point_4, place_point_4 = step_4_instructions(landmarks, landmarks_origional, fold_type=FOLD_TYPE)
    # Plot pickpoint on image
    cv2.circle(cv_image_4, tuple(pick_point_4.astype(int)), 10, (0, 255, 0), -1)
    cv2.circle(cv_image_4, tuple(place_point_4.astype(int)), 10, (255, 0, 0), -1)
    cv2.imshow("Image 4 with pick point", cv_image_4)
    cv2.waitKey(0)
    
    # get pick points for step 5
    pick_point_5, place_point_5 = step_5_instructions(landmarks, landmarks_origional, fold_type=FOLD_TYPE)
    # Plot pickpoint on image
    if FOLD_TYPE == 'square':
        cv2.circle(cv_image_5, tuple(pick_point_5.astype(int)), 10, (0, 255, 0), -1)
        cv2.circle(cv_image_5, tuple(place_point_5.astype(int)), 10, (255, 0, 0), -1)
        cv2.imshow("Image 5 with pick point", cv_image_5)
    cv2.imshow("Image 5", cv_image_5)
    cv2.waitKey(0)

    if FOLD_TYPE == 'star':
        return
    
    cv2.imshow("Image 6", cv_image_6)
    cv2.waitKey(0)


    
    
if __name__ == "__main__":
    main()