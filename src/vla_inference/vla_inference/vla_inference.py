#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import requests
import urllib3
import cv2
import sys

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class VlaInference(Node):
    def __init__(self, image_path):
        super().__init__('inference_client_node')

        # For now, just run once on startup
        self.get_logger().info("Reading image and sending to server...")
        self.send_image(image_path, "pick up the orange ball")

    def send_image(self, image_path, prompt : str):
        img = cv2.imread(image_path)
        if img is None:
            self.get_logger().error(f"Could not read image: {image_path}")
            return

        payload = {"data": img.flatten().tolist(), "prompt": prompt}

        try:
            res = requests.post("http://localhost:5000/", json=payload)
            
            self.get_logger().info(f"Status code: {res.status_code}")
            self.get_logger().info(f"Raw response: {res.text}")
            self.get_logger().info(f"JSON: {res.json()}")
            
        except Exception as e:
            self.get_logger().error(f"Request failed: {e}")


def main(args=None):
    rclpy.init(args=args)

    # Get image path from command line argument
    if len(sys.argv) < 2:
        print("Usage: ros2 run vla_inference inference_client_node <image_path>")
        return

    image_path = sys.argv[1]

    node = VlaInference(image_path)
    rclpy.spin_once(node)  # Run once
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
