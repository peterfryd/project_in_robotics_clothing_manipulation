import cv2
import numpy as np
import os


def load_correspondences(csv_path):
    """Load 2D-2D correspondences from CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Correspondence file not found: {csv_path}")
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    robot_points = 0.001 *data[:, 3:5]
    image_points = data[:, 1:3]
    return robot_points, image_points

def compute_homography(robot_points, image_points):
    """
    Compute a homography H that maps image_points -> robot_points.
    Uses the Direct Linear Transform (DLT) method.
    """
    robot_points = np.array(robot_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)
    H, _ = cv2.findHomography(image_points, robot_points, method=0)
    return H

def project_point(image_point, homography):
    image_point = image_point.astype(np.float64).reshape(1, 1, 2)
    projected = cv2.perspectiveTransform(image_point, homography)
    # Extract result
    x, y = projected[0, 0]

    return x,y


if __name__ == "__main__":
    # Generating homography
    csv_path = "data_points_new.csv"
    robot_points, image_points = load_correspondences(csv_path)
    H = compute_homography(robot_points, image_points)
    print("The generated homography is: \n", H)
    
    # Testing homography
    test_image_point = np.array([873, 628])
    x,y = project_point(test_image_point, H)
    print(f"\n The image point (u,v): ({test_image_point[0]}, {test_image_point[1]}) is mapped to the robot base at (x,y): ({x:.3f},{y:.3f})")
