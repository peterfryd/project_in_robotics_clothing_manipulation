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
        self.bridge = CvBridge()
        self.get_logger().info("Inference service ready")
    

    def inference(self, request, response):
        self.get_logger().info(f"Incoming request: prompt: {request.prompt}")

        cv_image = self.bridge.imgmsg_to_cv2(request.image)
        cv_image_cropped = cv_image[0:720, 280:1000]
        cv2.img_write("cropped.png", cv_image_cropped)  # Save the cropped image for debugging
        cv_image_resized = cv2.resize(cv_image_cropped, (224, 224))  # Resize to model's expected input size
        cv2.img_write("resized.png", cv_image_resized)
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