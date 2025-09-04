#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from PIL import Image as PILImage
import numpy as np
import cv2

class OpenVLANode(Node):
    def __init__(self):
        super().__init__('openvla_node')
        self.get_logger().info("OpenVLA Node started.")

        # Subscriber for RGB images
        self.subscription = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',  # Replace with your camera topic
            self.image_callback,
            10
        )

        self.bridge = CvBridge()

        # Load RT-2X model
        self.get_logger().info("Loading RT-2X model...")
        self.processor = AutoProcessor.from_pretrained("openvla/rt-2x")
        self.model = AutoModelForVision2Seq.from_pretrained("openvla/rt-2x").to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.get_logger().info("Model loaded.")

        # Example textual prompt for UR5e robot
        self.prompt = "UR5e robot: pick up the red cube and place it on the table."

    def image_callback(self, msg):
        # Convert ROS Image message to PIL image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        pil_image = PILImage.fromarray(cv_image)

        # Prepare inputs for the model
        inputs = self.processor(images=pil_image, text=self.prompt, return_tensors="pt").to(self.model.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
            text_output = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        self.get_logger().info(f"Inference result: {text_output}")

def main(args=None):
    rclpy.init(args=args)
    node = OpenVLANode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
