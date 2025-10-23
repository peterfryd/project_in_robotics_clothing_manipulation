#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from custom_interfaces_pkg.srv import GetPickAndPlacePoint
from image_processing_pkg.folding_instructions import load_background_image, step_1_instructions, step_2_instructions, step_3_instructions, step_4_instructions, step_5_instructions, step_6_instructions
from sensor_msgs.msg import Image
import threading
from cv_bridge import CvBridge


class GetPickAndPlacePointNode(Node):
    def __init__(self):
        super().__init__('get_pick_and_place_point_node')
        
        self.lock = threading.Lock()
        self.image = None
        self.bridge = CvBridge()
        
        self.background = load_background_image()
        
        self.sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )
        
        self.srv = self.create_service(
            GetPickAndPlacePoint,
            '/get_pick_and_place_point_srv',
            self.get_pick_and_place_point_handler
        )

        self.get_logger().info("get_pick_and_place_point node ready and providing /get_pick_and_place_point_srv service.")

    
    def image_callback(self, msg):
        with self.lock:
            self.image = msg
        
    def get_pick_and_place_point_handler(self, request, response):
        step_number = request.step_number
        cv_image = None
        
        with self.lock:          
            if self.image is None:
                self.get_logger().error("No image received yet.")
                response.image_pick_point = [0, 0]
                response.image_place_point = [0, 0]
                return response
            
            cv_image = self.bridge.imgmsg_to_cv2(self.image, desired_encoding='bgr8')

        step_functions = {
            1: step_1_instructions,
            2: step_2_instructions,
            3: step_3_instructions,
            4: step_4_instructions,
            5: step_5_instructions,
            6: step_6_instructions,
        }
        
        func = step_functions.get(step_number)
        if func is not None:
            image_pick_point, image_place_point = func(cv_image, self.background)
        else:
            self.get_logger().warn(f"Unknown step number {step_number}, returning [0, 0].")
            image_pick_point = [0, 0]
            image_place_point = [0, 0]

        response.image_pick_point = image_pick_point
        response.image_place_point = image_place_point

        self.get_logger().info(f"Returned pick and place point for step {step_number}: Pick: [{response.image_pick_point[0]:.3f}, {response.image_pick_point[1]:.3f}], Place [{response.image_place_point[0]:.3f}, {response.image_place_point[1]:.3f}]")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = GetPickAndPlacePointNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
