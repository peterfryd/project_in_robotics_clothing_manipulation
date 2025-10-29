from tkinter import image_names
import cv2
import numpy as np

def sift_track_points(img1, img2, points, ratio_thresh=0.7):
    """
    Track points from img1 to img2 using SIFT feature matching.

    Args:
        img1: Grayscale image at step1.
        img2: Grayscale image at step2.
        points: List of (x, y) points in img1 to track.
        ratio_thresh: Lowe's ratio test threshold.

    Returns:
        List of (x, y) points in img2 corresponding to input points.
        If a point cannot be matched, returns None for that point.
    """

    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Find the closest keypoints in img1 to the input points
    def closest_kp(point, keypoints):
        distances = [np.linalg.norm(np.array(kp.pt) - np.array(point)) for kp in keypoints]
        idx = np.argmin(distances)
        return keypoints[idx], idx

    # Match descriptors using FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

    # Map a point's index in kp1 to its match in kp2
    def match_point(old_idx, good_matches, kp2):
        for m in good_matches:
            if m.queryIdx == old_idx:
                return kp2[m.trainIdx].pt
        return None

    # Track each input point
    tracked_points = []
    for pt in points:
        kp_closest, idx_closest = closest_kp(pt, kp1)
        pt_in_img2 = match_point(idx_closest, good_matches, kp2)
        tracked_points.append(pt_in_img2)

    return tracked_points

def segment_foreground(image, background, visualize=False):
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
    if visualize:
        image_copy = image.copy()
        cv2.drawContours(image_copy, [central_contour], -1, (0, 255, 0), 10)

        cv2.imshow("Foreground Mask with Central image", forergound_mask)
        cv2.imshow("Clothing Contour Overlaid", image_copy)
        cv2.waitKey(0)

    return forergound_mask

# --- Load images ---z
#image_names = ["1_Color", "2_Color", "3_Color", "4_Color", "5_Color", "6_Color"]
image_names = ["1", "2", "3", "4", "5", "6"]
background = cv2.imread('images/background.png')
images = []
steps = []
instructions = []

for i in range(len(image_names)):
    image = cv2.imread(f'images/{image_names[i]}.png')
    #image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
    images.append(image)


# Convert to grayscale
for image in images:
    forergound_mask = segment_foreground(image, background, visualize=True)

# STEP 1

fg1, cc1 = steps[0]
# --- Convex hull of largest contour ---
hull_points = cv2.convexHull(cc1)  # Nx1x2
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
image_with_hull = fg1.copy()
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

M = cv2.moments(cc1)

if M["m00"] != 0:
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
else:
    cx, cy = 0, 0  # Avoid division by zero for degenerate contours

cv2.circle(image_with_hull, (cx, cy), 5, (255, 0, 0), -1)


for i in range(len(shirt_corners)):
    mid_vec = np.array([cx, cy]) - shirt_corners[i]
    pick_point= np.round(shirt_corners[i] + 20 * (mid_vec / np.linalg.norm(mid_vec))).astype(int)
    
    direction_vec = shirt_corners[(i+1)%2] - shirt_corners[i]
    place_point = np.round(shirt_corners[i] + 2/3 * direction_vec).astype(int)
    cv2.circle(image_with_hull, pick_point, 5, (255, 0, 0), -1)
    cv2.circle(image_with_hull, place_point, 5, (0, 0, 255), -1)
    instructions.append((tuple(pick_point), tuple(place_point)))

cv2.imshow('Foreground with Hexagon Hull', image_with_hull)
cv2.waitKey(0)

# STEP 2

fg2, cc2 = steps[1]

# Load images as grayscale
img0 = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY) 
img1 = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY) 

cv2.imshow('Step 1', img0)
cv2.imshow('Step 2', img1)
cv2.waitKey(0)
# Points you want to track from img1
points_step1 = [instructions[1][0]]  # pick and place points from step 1

# Track points to img2
points_step2 = sift_track_points(img0, img1, points_step1)

print("Mapped points in step2:", points_step2)

# Correct way to draw a circle
if points_step2[0] is not None:
    pt_int = (int(points_step2[0][0]), int(points_step2[0][1]))
    cv2.circle(fg2, pt_int, 5, (255, 0, 0), -1)  # pick point
    cv2.circle(fg2, instructions[1][1], 5, (0, 0, 255), -1)  # place point
    
instructions[1] = (pt_int, instructions[1][1])
cv2.imshow('Step 2 with Mapped Points', fg2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# STEP 3

image_3 = cv2.imread('images/steps/step3_masked.jpeg')
image_3 = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
images.append(image)