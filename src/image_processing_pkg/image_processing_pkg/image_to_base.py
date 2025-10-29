#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from custom_interfaces_pkg.srv import ImageToBase

class ImageToBaseNode(Node):
    def __init__(self):
        super().__init__('image_to_base_node')

        # Camera intrinsics
        self.fx = 908.2691650390625
        self.fy = 907.7402954101562
        self.cx = 637.5879516601562
        self.cy = 355.5464172363281
        
        # tcp to cam (mounting offset)
        self.R_tcp_to_cam = np.array([[0,1,0],
                                      [0,0,-1],
                                      [-1,0,0]])
        self.t_tcp_to_cam = np.array([-0.045, -0.06, -0.01])

        # base to tcp 
        self.R_base_to_tcp = self.axis_angle_to_rotation_matrix(0.973, -1.514, -1.506)
        self.t_base_to_tcp = np.array([-0.473, -0.230, 0.930])
        # joint pose = [33.37, -77.23, 37.00, 220.04, -147.22, -180.21]
        

        # Camera height to table
        self.camera_height = 0.928 - 0.06 + 0.035
        self.R_base_to_cam = self.R_base_to_tcp @ self.R_tcp_to_cam
        self.t_base_to_cam = self.R_base_to_tcp @ self.t_tcp_to_cam + self.t_base_to_tcp
        self.camera_height = self.t_base_to_cam[2] + 0.05
        
        # Service
        self.srv = self.create_service(
            ImageToBase,
            '/image_to_base_srv',
            self.handle_image_to_base
        )

        self.get_logger().info("image_to_base node ready and providing /image_to_base_srv service.")

    def handle_image_to_base(self, request, response):
        self.get_logger().info("Service called")

        u, v = request.imageframe_coordinates
        u, v = float(u), float(v)

        Z_cam = self.camera_height
        #u, v = 640, 360

        X_cam = (u - self.cx) * Z_cam / self.fx
        Y_cam = (v - self.cy) * Z_cam / self.fy

        P_cam = np.array([X_cam, Y_cam, Z_cam])

        self.get_logger().info(f"Projected ({u}, {v}) to P_cam: [{P_cam[0]:.3f}, {P_cam[1]:.3f}, {P_cam[2]:.3f}]")
        
        # Transformation from camera to TCP
        P_tcp = self.R_tcp_to_cam @ P_cam + self.t_tcp_to_cam

        # Transformation from TCP to base
        P_base = self.R_base_to_tcp @ P_tcp + self.t_base_to_tcp

        response.baseframe_coordinates = P_base.tolist()

        self.get_logger().info(f"Transformed ({u}, {v}) to P_base: [{P_base[0]:.3f}, {P_base[1]:.3f}, {P_base[2]:.3f}]")

        return response
    
    def axis_angle_to_rotation_matrix(self, r1, r2, r3):
        r = np.array([r1, r2, r3])
        angle = np.linalg.norm(r)
        if angle < 1e-8:  # near zero angle
            return np.eye(3)

        axis = r / angle
        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c

        R = np.array([
            [t*x*x + c,     t*x*y - s*z, t*x*z + s*y],
            [t*x*y + s*z,   t*y*y + c,   t*y*z - s*x],
            [t*x*z - s*y,   t*y*z + s*x, t*z*z + c]
        ])
        return R


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