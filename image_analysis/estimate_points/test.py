import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# Helper functions
# -----------------------------
def load_pixel_coordinates(csv_path):
    return pd.read_csv(csv_path).values

def load_ground_truth_plane(csv_path):
    return pd.read_csv(csv_path).values

def apply_homography(points, H):
    # points: Nx2 pixel coordinates
    # H: 3x3 homography
    pts = np.hstack([points, np.ones((points.shape[0], 1))])
    proj = (H @ pts.T).T
    proj /= proj[:, 2].reshape(-1, 1)
    return proj[:, :3]  # x, y, 1 (but returned as 3D for plotting)

def apply_projection_matrix(points, P):
    # points: Nx2 pixel coordinates
    # P: 3x4 projection matrix
    pts = np.hstack([points, np.zeros((points.shape[0], 1)), np.ones((points.shape[0], 1))])
    proj = (P @ pts.T).T
    proj /= proj[:, 2].reshape(-1, 1)
    return proj[:, :3]

# -----------------------------
# Main visualization pipeline
# -----------------------------
def compare_projections(pixel_csv, plane_csv, homography, projection_matrix):
    # Load input data
    pixels = load_pixel_coordinates(pixel_csv)
    plane = load_ground_truth_plane(plane_csv)

    # Compute projections
    proj_h = apply_homography(pixels, homography)
    proj_p = apply_projection_matrix(pixels, projection_matrix)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Ground truth plane
    ax.plot_trisurf(plane[:,0], plane[:,1], plane[:,2], alpha=0.4)

    # Projected points
    ax.scatter(proj_h[:,0], proj_h[:,1], proj_h[:,2], label='Homography', s=20)
    ax.scatter(proj_p[:,0], proj_p[:,1], proj_p[:,2], label='Projection Matrix', s=20)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Replace with your CSV paths
    pixel_csv_path = "pixels.csv"
    plane_csv_path = "plane.csv"

    # Replace with your matrices
    H = np.array([[9.03593152e-04, -4.26739325e-04, -8.48637901e-01],
                  [-4.15489185e-04, -9.06983303e-04,  4.25934133e-01],
                  [-2.13676908e-05,  2.56016933e-05,  1.00000000e+00]])
    
    
    fx = 908.2691650390625
    fy = 907.7402954101562
    cx = 637.5879516601562
    cy = 355.5464172363281
    
    # tcp to cam (mounting offset)
    R_tcp_to_cam = np.array([[0,1,0],
                                    [0,0,-1],
                                    [-1,0,0]])
    t_tcp_to_cam = np.array([-0.045, -0.055, -0.01])

    # base to tcp 
    R_base_to_tcp = axis_angle_to_rotation_matrix(0.973, -1.514, -1.506)
    t_base_to_tcp = np.array([-0.473, -0.230, 0.930])
    # joint pose = [33.37, -77.23, 37.00, 220.04, -147.22, -180.21]
    
    H = np.array([[9.03593152e-04, -4.26739325e-04, -8.48637901e-01],
                        [-4.15489185e-04, -9.06983303e-04,  4.25934133e-01],
                        [-2.13676908e-05,  2.56016933e-05,  1.00000000e+00]])
    
    z_height = 0.0
    
    # Camera height to table
    camera_height = 0.928 - 0.06 + 0.035
    R_base_to_cam = R_base_to_tcp @ R_tcp_to_cam
    t_base_to_cam = R_base_to_tcp @ t_tcp_to_cam + t_base_to_tcp
    camera_height = t_base_to_cam[2] + 0.05
    P = np.hstack([np.eye(3), np.zeros((3,1))])

    compare_projections(pixel_csv_path, plane_csv_path, H, P)
