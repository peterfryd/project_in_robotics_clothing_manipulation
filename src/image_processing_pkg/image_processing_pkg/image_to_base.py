#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from custom_interfaces_pkg.srv import ImageToBase

class ImageToBaseNode(Node):
    def __init__(self):
        super().__init__('image_to_base_node')

        # Camera height to table
        self.camera_height = 1.0

        # Camera intrinsics
        self.fx = 891.01611328125
        self.fy = 891.01611328125
        self.cx = 642.7546997070312
        self.cy = 367.76971435546875
        
        # tcp to base transform
        self.R_tcp_to_base = self.rotation_matrix_from_euler_xyz(np.pi/2, 0.0, -1.992)
        self.t_tcp_to_base = np.array([-0.473, -0.230, 0.530])
        # joint pose = [33.37, -77.23, 37.00, 220.04, -147.22, -180.21]
        
        # cam to tcp transform (mounting offset)
        self.R_cam_to_tcp = self.rotation_matrix_from_euler_xyz(np.pi/2, 0.0, 0.0)
        self.t_cam_to_tcp = np.array([-0.045, -0.06, -0.01]) 

        # Service
        self.srv = self.create_service(
            ImageToBase,
            '/image_to_base_srv',
            self.handle_image_to_base
        )

        self.get_logger().info("image_to_base node ready and providing /image_to_base_srv service.")

    def handle_image_to_base(self, request, response):
        u, v = request.imageframe_coordinates

        Z_cam = self.camera_height
        Z_cam = 2.047
        u, v = 560, 350
        X_cam = (u - self.cx) * Z_cam / self.fx
        Y_cam = (v - self.cy) * Z_cam / self.fy

        P_cam = np.array([X_cam, Y_cam, Z_cam])
        
        # Transformation from camera to TCP
        P_tcp = self.R_cam_to_tcp @ P_cam + self.t_cam_to_tcp

        # Transformation from TCP to base
        P_base = self.R_tcp_to_base @ P_tcp + self.t_tcp_to_base

        response.baseframe_coordinates = P_base.tolist()

        self.get_logger().info(f"Projected ({u}, {v}) -> [{P_base[0]:.3f}, {P_base[1]:.3f}, {P_base[2]:.3f}]")

        return response

    def rotation_matrix_from_euler_xyz(self, roll, pitch, yaw):
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
        return Rz @ Ry @ Rx


def main(args=None):
    rclpy.init(args=args)
    node = ImageToBaseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
