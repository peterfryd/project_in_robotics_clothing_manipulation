
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


def step_1_instructions(cv_image, background):
    foregroundd_mask, central_contour = segment_foreground(cv_image, background)
    
    hull_points = cv2.convexHull(central_contour)
    hull_points = hull_points.reshape(-1, 2)

    epsilon = 0.01 * cv2.arcLength(hull_points, True)
    for i in range(100):
        approx = cv2.approxPolyDP(hull_points, epsilon, True)
        if len(approx) <= 4:
            break
        epsilon *= 1.05

    if len(approx) < 4:
        approx = approx[np.linspace(0, len(approx)-1, 6, dtype=int)]

    rectangle = approx.reshape(-1, 2)

    image_with_hull = cv_image.copy()
    distances = []
    for i in range(4):
        pt1 = tuple(rectangle[i])
        pt2 = tuple(rectangle[(i + 1) % 4])
        distances.append(float(np.linalg.norm(np.array(pt1) - np.array(pt2))))
        cv2.line(image_with_hull, pt1, pt2, (0, 0, 255), 2)

    point_occurences = {}

    indices = np.argsort(np.array(distances))[-3:]
    for idx in indices:
        pt1 = tuple(rectangle[idx])
        pt2 = tuple(rectangle[(idx + 1) % 4])
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
        cx, cy = 0, 0

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
        
    cv2.imwrite('/home/anders/Pictures/robot_images/image_1_instructions.png', image_with_hull)
    
    return pick_point, place_point
    

def step_2_instructions(cv_image, background, step_1_place):
    foreground_mask, central_contour = segment_foreground(cv_image, background)
    
    hull_points = cv2.convexHull(central_contour)
    hull_points = hull_points.reshape(-1, 2)

    epsilon = 0.01 * cv2.arcLength(hull_points, True)
    for i in range(100):
        approx = cv2.approxPolyDP(hull_points, epsilon, True)
        if len(approx) <= 4:
            break
        epsilon *= 1.05

    if len(approx) < 4:
        approx = approx[np.linspace(0, len(approx)-1, 6, dtype=int)]

    rectangle = approx.reshape(-1, 2)

    image_with_hull = cv_image.copy()
    
    min_dist = inf
    pick_point = [0, 0]
    
    for pt in rectangle:
        dist = np.linalg.norm(pt - step_1_place)
        if dist < min_dist:
            min_dist = dist
            pick_point = pt
    
    remaining_pts = np.array([pt for pt in rectangle if not np.allclose(pt, pick_point)])
    max_dist = -inf
    
    for i in range(len(remaining_pts)):
        for j in range(i + 1, len(remaining_pts)):
            dist = np.linalg.norm(remaining_pts[i] - remaining_pts[j])
            if dist > max_dist:
                max_dist = dist
                p1, p2 = remaining_pts[i], remaining_pts[j]

    p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)

    v = p2 - p1
    place_point = np.astype(p1 + (p2-p1) / 2, np.int32).tolist()
    
    cv2.circle(image_with_hull, place_point, 5, (0, 0, 255), -1)
    cv2.circle(image_with_hull, pick_point, 5, (255, 0, 0), -1)
        
    cv2.imwrite('/home/anders/Pictures/robot_images/image_2_instructions.png', image_with_hull)
    
    return pick_point, place_point

def step_3_instructions(cv_image, background):
    foreground_mask, central_contour = segment_foreground(cv_image, background)
    
    hull_points = cv2.convexHull(central_contour)
    hull_points = hull_points.reshape(-1, 2)

    epsilon = 0.01 * cv2.arcLength(hull_points, True)
    for i in range(100):
        approx = cv2.approxPolyDP(hull_points, epsilon, True)
        if len(approx) <= 3:
            break
        epsilon *= 1.05

    if len(approx) < 3:
        approx = approx[np.linspace(0, len(approx)-1, 6, dtype=int)]

    triangle = approx.reshape(-1, 2)

    image_with_hull = cv_image.copy()
    min_dist = inf
    pt1 = [0,0]
    pt2 = [0,0]
    
    for i in range(len(triangle)):
        for j in range(i + 1, len(triangle)):
            dist = np.linalg.norm(triangle[i] - triangle[j])
            if dist < min_dist:
                min_dist = dist
                p1, p2 = triangle[i], triangle[j]
    
    pick_point = p2
    place_point = np.astype(p2 + 0.5 * (p1 - p2), np.int32)

    cv2.circle(image_with_hull, place_point, 5, (0, 0, 255), -1)
    cv2.circle(image_with_hull, pick_point, 5, (255, 0, 0), -1)
        
    cv2.imwrite('/home/anders/Pictures/robot_images/image_3_instructions.png', image_with_hull)
    
    return pick_point, place_point

def step_4_instructions(cv_image, background, step_3_pick, step_3_place):
    foreground_mask, central_contour = segment_foreground(cv_image, background)
    
    hull_points = cv2.convexHull(central_contour)
    hull_points = hull_points.reshape(-1, 2)

    epsilon = 0.01 * cv2.arcLength(hull_points, True)
    for i in range(100):
        approx = cv2.approxPolyDP(hull_points, epsilon, True)
        if len(approx) <= 4:
            break
        epsilon *= 1.05

    if len(approx) < 4:
        approx = approx[np.linspace(0, len(approx)-1, 6, dtype=int)]

    rectangle = approx.reshape(-1, 2)

    image_with_hull = cv_image.copy()
    
    place_point = [0, 0]
    min_dist_pick = inf
    
    for pt in rectangle:
        dist_pick = np.linalg.norm(pt - step_3_pick)
        if dist_pick < min_dist_pick:
            min_dist_pick = dist_pick
            place_point = pt.tolist()
    
    pick_point = [0, 0]
    
    remaining_pts = np.array([pt for pt in rectangle if not np.allclose(pt, place_point)])
    
    pick_point = [0, 0]
    min_dist_place = inf
    
    for pt in remaining_pts:
        dist_place = np.linalg.norm(pt - step_3_place)
        if dist_pick < min_dist_place:
            min_dist_place = dist_place
            pick_point = pt.tolist()
    
    cv2.circle(image_with_hull, place_point, 5, (0, 0, 255), -1)
    cv2.circle(image_with_hull, pick_point, 5, (255, 0, 0), -1)
        
    cv2.imwrite('/home/anders/Pictures/robot_images/image_4_instructions.png', image_with_hull)
    return pick_point, place_point

def step_5_instructions(cv_image, background, place_point_4):
    foregroundd_mask, central_contour = segment_foreground(cv_image, background)
    
    hull_points = cv2.convexHull(central_contour)
    hull_points = hull_points.reshape(-1, 2)

    epsilon = 0.01 * cv2.arcLength(hull_points, True)
    for i in range(100):
        approx = cv2.approxPolyDP(hull_points, epsilon, True)
        if len(approx) <= 4:
            break
        epsilon *= 1.05

    if len(approx) < 4:
        approx = approx[np.linspace(0, len(approx)-1, 6, dtype=int)]

    rectangle = approx.reshape(-1, 2)

    image_with_hull = cv_image.copy()
    
    min_dist_pick = inf
    
    pt1 = None
    pt2 = None
    
    for pt in rectangle:
        cv2.circle(image_with_hull, pt, 5, (0, 255, 255), -1)
        dist_pick = np.linalg.norm(pt - place_point_4)
        if dist_pick < min_dist_pick:
            min_dist_pick = dist_pick
            pt1 = pt
    
    
    remaining_pts = np.array([pt for pt in rectangle if not np.allclose(pt, pt1)])
    
    pick_point = [0, 0]
    min_dist_place = inf
    
    for pt in remaining_pts:
        dist_place = np.linalg.norm(pt - pt1)
        if dist_pick < min_dist_place:
            min_dist_place = dist_place
            pt2 = pt
    
    mid_point_pick = (pt2 + pt1) / 2
    
    rectangle_mid = np.mean(rectangle, axis=0)
    
    pick_point =(mid_point_pick + 50 * (rectangle_mid - mid_point_pick)/(np.linalg.norm(rectangle_mid - mid_point_pick)))
    
    
    last_two_points  = np.array([pt for pt in remaining_pts if not np.allclose(pt, pt2)])
    
    place_point = np.mean(last_two_points, axis = 0)
    
    
    rectangle_mid = np.astype(rectangle_mid, np.int32)
    pick_point = np.astype(pick_point, np.int32)
    place_point = np.astype(place_point, np.int32)
    
    cv2.circle(image_with_hull, rectangle_mid.tolist(), 5, (0, 0, 255), -1)
    cv2.circle(image_with_hull, place_point, 5, (0, 0, 255), -1)
    cv2.circle(image_with_hull, pick_point, 5, (255, 0, 0), -1)
        
    cv2.imwrite('/home/anders/Pictures/robot_images/image_5_instructions.png', image_with_hull)
    return pick_point, place_point
    



def main():
    cv_image_1 = cv2.imread("/home/peter/uni/project_clothing_fresh/image_analysis/images/1.png")
    cv_image_2 = cv2.imread("/home/peter/uni/project_clothing_fresh/image_analysis/images/2.png")
    cv_image_3 = cv2.imread("/home/peter/uni/project_clothing_fresh/image_analysis/images/3.png")
    cv_image_4 = cv2.imread("/home/peter/uni/project_clothing_fresh/image_analysis/images/4.png")
    cv_image_5 = cv2.imread("/home/peter/uni/project_clothing_fresh/image_analysis/images/5.png")
    
    back_ground = cv2.imread("/home/peter/uni/project_clothing_fresh/image_analysis/images/background.png")
    
    pick_point_1, place_point_1 = step_1_instructions(cv_image_1, back_ground)
    pick_point_2, place_point_2 = step_2_instructions(cv_image_2, back_ground, pick_point_1)
    pick_point_3, place_point_3 = step_3_instructions(cv_image_3, back_ground)
    pick_point_4, place_point_4 = step_4_instructions(cv_image_4, back_ground, pick_point_3, place_point_3)
    pick_point_5, place_point_5 = step_5_instructions(cv_image_5, back_ground, place_point_4)
    
if __name__ == "__main__":
    main()