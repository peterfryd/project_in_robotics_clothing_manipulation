#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from custom_interfaces_pkg.srv import GetLandmarks
from sensor_msgs.msg import Image
import threading
from cv_bridge import CvBridge
import cv2
import os



class GetLandmarksNode(Node):
    def __init__(self):
        super().__init__('get_pick_and_place_point_node')

        self.lock = threading.Lock()
        self.image = None
        self.bridge = CvBridge()
        
        self.background = None
        
        self.sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )
        
        self.srv = self.create_service(
            GetLandmarks,
            '/get_landmarks_srv',
            self.get_landmarks
        )

        self.get_logger().info("get_pick_and_place_point node ready and providing /get_pick_and_place_point_srv  and /update_background_image service.")

    
    def image_callback(self, msg):
        with self.lock:
            self.image = msg
    
    def get_landmarks(self, request, response):
        cv_image = None
        with self.lock:
            if self.image is not None:
                cv_image = self.bridge.imgmsg_to_cv2(self.image, desired_encoding='bgr8')
            else:
                self.get_logger().error("No Image available to find landmarks")
                
        response.landmarks = self.run_inference(cv_image)
        return response
        
    def run_inference(self, image):
        # Placeholder for actual inference code
        # This function should return the landmarks detected in the image
        landmarks = {
            'top_left': (50, 50),
            'top_right': (150, 50),
            'bottom_left': (50, 150),
            'bottom_right': (150, 150)
        }
        return landmarks

def main(args=None):
    rclpy.init(args=args)
    node = GetLandmarksNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
