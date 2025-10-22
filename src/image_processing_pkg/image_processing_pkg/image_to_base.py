#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
from custom_interfaces_pkg.srv import ImageToBase

class JazzyNode(Node):
    def __init__(self):
        super().__init__('jazzy_node')

        # Latest depth image and camera intrinsics
        self.latest_depth_image = None
        self.fx = self.fy = self.cx = self.cy = None

        self.bridge = CvBridge()

        # Subscribers
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/depth/camera_info',
            self.camera_info_callback,
            10
        )

        # Service
        self.srv = self.create_service(
            ImageToBase,
            'image_to_base',
            self.handle_image_to_base
        )

        self.get_logger().info("JazzyNode ready: listening to RealSense and providing ImageToBase service.")

    def depth_callback(self, msg: Image):
        self.latest_depth_image = msg

    def camera_info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def handle_image_to_base(self, request, response):
        if self.latest_depth_image is None or None in [self.fx, self.fy, self.cx, self.cy]:
            self.get_logger().warn("Depth image or camera info not ready!")
            response.baseframe_coordinates = [float('nan')] * 3
            return response

        u, v = request.imageframe_coordinates

        # Convert ROS image to NumPy array
        depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth_image, desired_encoding='passthrough')
        depth_value = (depth_image[int(v), int(u)]) / 1000 # Z in mm (check your RealSense depth scaling)

        self.get_logger().info(f"Depth value at ({u:.1f}, {v:.1f}): {depth_value:.5f}")

        # Projection math
        X = (u - self.cx) * depth_value / self.fx
        Y = (v - self.cy) * depth_value / self.fy
        Z = depth_value

        response.baseframe_coordinates = [X, Y, Z]
        self.get_logger().info(f"Projected ({u}, {v}) -> [{X:.3f}, {Y:.3f}, {Z:.3f}]")

        return response


def main(args=None):
    rclpy.init(args=args)
    node = JazzyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
