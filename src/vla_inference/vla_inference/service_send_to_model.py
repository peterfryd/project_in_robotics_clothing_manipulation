from vla_inference.srv import ServiceSendToModel
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node

import requests
import cv2


class MinimalService(Node):

    def __init__(self):
        super.__init__('Minimal_service')
        self.srv =  self.create_service(ServiceSendToModel, "service_send_to_model", self.send_to_model_callback)
    

    def send_to_model_callback(self, request, response):
        self.get_logger().info(f"Incoming request: prompt: {request.prompt}")

        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(request.image)

        payload = {"data": cv_image.flatten().tolist(), "prompt": request.prompt}

        try:
            res = requests.post("http://localhost:5000/", json=payload)            
        except Exception as e:
            self.get_logger().error(f"Request failed: {e}")

        response.delta_position = res[0:3]
        response.delta_orientation = res[3:6]
        response.delta_gripper = res[6]

        return response