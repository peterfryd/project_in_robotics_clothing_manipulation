from custom_interfaces_pkg.srv import Inference
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node

import requests
import cv2

class ModelInference(Node):

    def __init__(self):
        super().__init__('inference')
        self.srv =  self.create_service(Inference, "inference_srv", self.inference)
    

    def inference(self, request, response):
        self.get_logger().info(f"Incoming request: prompt: {request.prompt}")

        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(request.image)

        cv_image_resized = cv2.resize(cv_image, (224, 224))  # Resize to model's expected input size
        self.get_logger().info("Image resized, sending to model...")
        
        payload = {"data": cv_image_resized.flatten().tolist(), "prompt": request.prompt}

        try:
            res = requests.post("http://localhost:5000/", json=payload)
            data = res.json()
        
            response.delta_position = data['result'][0:3]
            response.delta_orientation = data['result'][3:6]
            response.delta_gripper = data['result'][6]            
        except Exception as e:
            self.get_logger().error(f"Request failed: {e}")

        self.get_logger().info("Response prepared, sending back to client")
        return response
    
    
def main(args=None):
    rclpy.init(args=args)

    node = ModelInference()
    try:
        rclpy.spin(node)  # Keep node alive to handle incoming requests
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()