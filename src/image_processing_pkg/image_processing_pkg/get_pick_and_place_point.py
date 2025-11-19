#!/usr/bin/env python3

from std_srvs.srv import Empty
import rclpy
from rclpy.node import Node
import numpy as np
from custom_interfaces_pkg.srv import GetPickAndPlacePoint
from image_processing_pkg.folding_instructions import load_background_image, step_1_instructions, step_2_instructions, step_3_instructions, step_4_instructions, step_5_instructions
from sensor_msgs.msg import Image
import threading
from cv_bridge import CvBridge
import cv2
import os
from ament_index_python.packages import get_package_share_directory


class GetPickAndPlacePointNode(Node):
    def __init__(self):
        super().__init__('get_pick_and_place_point_node')
        
        self.place_point1 = None
        self.place_point3 = None
        self.pick_point3 = None
        self.place_point4 = None
        
        self.pkg_path = get_package_share_directory('image_processing_pkg')
        
        self.lock = threading.Lock()
        self.image_to_use = None
        self.image_newest = None
        self.bridge = CvBridge()
        
        self.background = None
        
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

        # self.back_ground_srv = self.create_service(
        #     Empty,
        #     '/update_background_image_srv',
        #     self.update_back_ground_image
        # )

        self.get_logger().info("get_pick_and_place_point node ready and providing /get_pick_and_place_point_srv  and /update_background_image service.")

    
    def image_callback(self, msg):
        with self.lock:
            self.image_newest = msg
    
    # def update_back_ground_image(self, request, response):
    #     cv_image = None
    #     with self.lock:
    #         if self.image is not None:
    #             cv_image = self.bridge.imgmsg_to_cv2(self.image, desired_encoding='bgr8')
    #         else:
    #             self.get_logger().error("No Image available to update the background image")
    #             return response
            
    #     mask_path = os.path.join(self.pkg_path, 'data', "mask.png")
        
    #     mask_image = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        
    #     mask = cv2.bitwise_not(cv2.inRange(mask_image, np.array([255, 255, 255]), np.array([255, 255, 255])))
        
    #     background = cv2.bitwise_and(cv_image, cv_image, mask=mask)
        
    #     with self.lock:
    #         self.background = background
        
    #     self.get_logger().info("Background image updated.")
    #     return response
            
    
    def get_pick_and_place_point_handler(self, request, response):
        step_number = request.step_number
        landmarks = request.landmarks
        cv_image = None
        
        # Check if image is ready
        with self.lock:
            if step_number == 1:
                if self.image_newest is None:
                    self.get_logger().error("No image available to process for step 1.")
                    response.image_pick_point = [0, 0]
                    response.image_place_point = [0, 0]
                    return response
                else:
                    self.image_to_use = self.image_newest
            elif self.image_to_use is None:
                self.get_logger().error("No image saved from previous steps to process.")
                response.image_pick_point = [0, 0]
                response.image_place_point = [0, 0]
                return response
            
            cv_image = self.bridge.imgmsg_to_cv2(self.image_to_use, desired_encoding='bgr8')
        
        pick_point = [0, 0]
        place_point = [0, 0]
        
        if step_number == 1:
            self.get_logger().info("Running Step 1")
            pick_point, place_point = step_1_instructions(landmarks=landmarks)
            self.place_point1 = place_point
        # elif step_number == 2:
        #     if self.place_point1 is None:
        #         self.get_logger().warn("Run step 1 before running step 2")
        #     else:
        #         self.get_logger().info("Running Step 2")
        #         pick_point, place_point = step_2_instructions(cv_image, self.background, self.place_point1)
        # elif step_number == 3:
        #     self.get_logger().info("Running Step 3")
        #     pick_point, place_point = step_3_instructions(cv_image, self.background)
        #     self.place_point3 = place_point
        #     self.pick_point3 = pick_point
        # elif step_number == 4:
        #     if self.place_point3 is None or self.pick_point3 is None:
        #         self.get_logger().warn("Run step 3 before running step 4")
        #     else:
        #         self.get_logger().info("Running Step 4")
        #         pick_point, place_point = step_4_instructions(cv_image, self.background, self.pick_point3, self.place_point3)
        #         self.place_point4 = place_point
        # elif step_number == 5:
        #     self.get_logger().info("Running Step 5")
        #     pick_point, place_point = step_5_>instructions(cv_image, self.background, self.place_point4)

        # else:
        #     self.get_logger().warn(f"Unknown step number {step_number}, returning [0, 0].")
            

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
