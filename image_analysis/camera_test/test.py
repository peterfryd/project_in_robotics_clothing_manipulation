import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# Helper functions (Uændret, undtagen set_3d_lims)
# -----------------------------
def create_ground_truth_plane(x_min, x_max, y_min, y_max, z_value, num_points):
    """Simulerer et ground truth plan i 3D."""
    xs = np.linspace(x_min, x_max, num_points)
    ys = np.linspace(y_min, y_max, num_points)
    grid_x, grid_y = np.meshgrid(xs, ys)
    plane = np.vstack([grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, z_value)]).T
    return plane

def apply_homography(points, H):
    """Projekterer 2D pixels til 3D World (Z=0 plane) via Homografi."""
    pts = np.hstack([points, np.ones((points.shape[0], 1))])
    proj = (H @ pts.T).T
    proj /= proj[:, 2].reshape(-1, 1)
    return np.hstack([proj[:, :2], np.zeros((points.shape[0], 1))])

def apply_projection_matrix(points, P):
    """Projekterer 2D pixels til 3D World (Z=0 plane) via invers af Homografi udledt fra P."""
    H_P = P[:, [0, 1, 3]]
    H_P_inv = np.linalg.inv(H_P)
    pts = np.hstack([points, np.ones((points.shape[0], 1))])
    proj = (H_P_inv @ pts.T).T
    proj /= proj[:, 2].reshape(-1, 1)
    return np.hstack([proj[:, :2], np.zeros((points.shape[0], 1))])


def compute_projection_matrix(fx, fy, cx, cy,
                              R_tcp_to_cam, t_tcp_to_cam,
                              R_base_to_tcp, t_base_to_tcp):
    """Beregner den 3x4 Projektionsmatrix P."""
    K = np.array([[fx, 0, cx], [0, fy, cy], [0,  0,  1]])
    R_base_to_cam = R_base_to_tcp @ R_tcp_to_cam
    t_base_to_cam = R_base_to_tcp @ t_tcp_to_cam + t_base_to_tcp
    R_world_to_cam = R_base_to_cam.T
    t_world_to_cam = -R_base_to_cam.T @ t_base_to_cam
    Rt = np.hstack([R_world_to_cam, t_world_to_cam.reshape(3,1)])
    P = K @ Rt
    return P

# -----------------------------
# Opdateret Skaleringsfunktion
# -----------------------------
def set_3d_lims(ax, proj_points, plane):
    """
    Sikrer ensartet skalering på 3D-plottet ved at inkludere 
    både projektionspunkter og ground truth planet.
    """
    # Sikrer at vi har data
    if proj_points.size == 0 or plane.size == 0:
        return

    # Kombiner data fra de projicerede punkter og ground truth planet
    combined_points = np.vstack([proj_points, plane])
        
    x, y, z = combined_points[:,0], combined_points[:,1], combined_points[:,2]
    
    # Beregn spændvidden over alle kombinerede punkter
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    
    # Beregn midtpunkter
    mid_x = (x.max()+x.min()) / 2.0
    mid_y = (y.max()+y.min()) / 2.0
    mid_z = (z.max()+z.min()) / 2.0
    
    # Sæt aksebegrænsninger
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def plot_projection(proj_points, plane, title, color):
    """Genererer et 3D-plot for en specifik projektion."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Ground truth plane visualization
    ax.plot_trisurf(plane[:,0], plane[:,1], plane[:,2], 
                    color='gray', alpha=0.2, label='Simuleret Ground Truth Plan (Z=0)')

    # Projected points
    ax.scatter(proj_points[:,0], proj_points[:,1], proj_points[:,2], 
               label=title, s=5, c=color)

    ax.set_xlabel('X (Base Frame) [m]')
    ax.set_ylabel('Y (Base Frame) [m]')
    ax.set_zlabel('Z (Base Frame) [m]')
    ax.set_title(title)
    ax.legend()
    
    # Sæt aksebegrænsninger ved at inkludere både proj. punkter og plane
    set_3d_lims(ax, proj_points, plane) 
    plt.show()

# -----------------------------
# Main visualization pipeline (Uændret)
# -----------------------------
def compare_projections(image_width, image_height, step,
                        H,
                        fx, fy, cx, cy,
                        R_tcp_to_cam, t_tcp_to_cam,
                        R_base_to_tcp, t_base_to_tcp):

    xs = np.arange(0, image_width, step)
    ys = np.arange(0, image_height, step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pixels = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    P = compute_projection_matrix(fx, fy, cx, cy,
                                  R_tcp_to_cam, t_tcp_to_cam,
                                  R_base_to_tcp, t_base_to_tcp)

    base_x, base_y, base_z = t_base_to_tcp
    # Gør planet lidt større for bedre visualisering
    plane = create_ground_truth_plane(base_x - 0.5, base_x + 0.5, 
                                      base_y - 0.5, base_y + 0.5, 
                                      0.0, 50) 
    
    # 1. Homografi Projektion
    proj_h = apply_homography(pixels, H)
    plot_projection(proj_h, plane, 
                    title='1. Homografi Projektion af Pixels til Z=0 Plan (H)', 
                    color='red')

    # 2. Projektionsmatrix (P) Projektion
    proj_p = apply_projection_matrix(pixels, P)
    plot_projection(proj_p, plane, 
                    title='2. Projektionsmatrix (P) Projektion af Pixels til Z=0 Plan', 
                    color='blue')


# -----------------------------
# Example usage (Uændret)
# -----------------------------
if __name__ == "__main__":
    image_width = 1280
    image_height = 720
    step = 100

    fx = 908.2691650390625
    fy = 907.7402954101562
    cx = 637.5879516601562
    cy = 355.5464172363281

    H = np.eye(3) 

    R_tcp_to_cam = np.array([[0,1,0],[0,0,-1],[-1,0,0]])
    t_tcp_to_cam = np.array([-0.045, -0.055, -0.01])

    def axis_angle_to_rotation_matrix(r1, r2, r3):
        r = np.array([r1, r2, r3])
        angle = np.linalg.norm(r)
        if angle < 1e-8:
            return np.eye(3)
        axis = r / angle
        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c
        return np.array([
            [t*x*x + c,     t*x*y - s*z, t*x*z + s*y],
            [t*x*y + s*z,   t*y*y + c,   t*y*z - s*x],
            [t*x*z - s*y,   t*y*z + s*x, t*z*z + c]
        ])

    R_base_to_tcp = axis_angle_to_rotation_matrix(0.973, -1.514, -1.506)
    t_base_to_tcp = np.array([-0.473, -0.230, 0.930])

    compare_projections(image_width, image_height, step,
                        H,
                        fx, fy, cx, cy,
                        R_tcp_to_cam, t_tcp_to_cam,
                        R_base_to_tcp, t_base_to_tcp)