#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from custom_interfaces_pkg.srv import GetPickAndPlacePoint
from image_processing_pkg.folding_instructions import load_background_image, step_1_instructions, step_2_instructions, step_3_instructions, step_4_instructions, step_5_instructions
from sensor_msgs.msg import Image
import threading
from cv_bridge import CvBridge

class GetPickAndPlacePointNode(Node):
    def __init__(self):
        super().__init__('get_pick_and_place_point_node')
        
        self.place_point1 = None
        self.place_point3 = None
        self.pick_point3 = None
        self.place_point4 = None
        
        self.lock = threading.Lock()
        self.image = None
        self.bridge = CvBridge()
        
        self.background = load_background_image(image_name = 'background.png')
        
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
        
        pick_point = [0, 0]
        place_point = [0, 0]
        
        if step_number == 1:
            self.get_logger().info("Running Step 1")
            pick_point, place_point = step_1_instructions(cv_image, self.background)
            self.place_point1 = place_point
        elif step_number == 2:
            if self.place_point1 is None:
                self.get_logger().warn("Run step 1 before running step 2")
            else:
                self.get_logger().info("Running Step 2")
                pick_point, place_point = step_2_instructions(cv_image, self.background, self.place_point1)
        elif step_number == 3:
            self.get_logger().info("Running Step 3")
            pick_point, place_point = step_3_instructions(cv_image, self.background)
            self.place_point3 = place_point
            self.pick_point3 = pick_point
        elif step_number == 4:
            if self.place_point3 is None or self.pick_point3 is None:
                self.get_logger().warn("Run step 3 before running step 4")
            else:
                self.get_logger().info("Running Step 4")
                pick_point, place_point = step_4_instructions(cv_image, self.background, self.pick_point3, self.place_point3)
                self.place_point4 = place_point
        elif step_number == 5:
            self.get_logger().info("Running Step 5")
            pick_point, place_point = step_5_instructions(cv_image, self.background, self.place_point4)

        else:
            self.get_logger().warn(f"Unknown step number {step_number}, returning [0, 0].")
            

        response.image_pick_point = pick_point
        response.image_place_point = place_point

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
