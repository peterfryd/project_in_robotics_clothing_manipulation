#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from PIL import Image as PILImage
import os

class OpenVLANode(Node):
    def __init__(self, image_folder):
        super().__init__('openvla_node')
        self.get_logger().info("OpenVLA Node started.")

        # Path to folder with images
        self.image_folder = image_folder
        self.image_files = sorted([os.path.join(image_folder, f) 
                                   for f in os.listdir(image_folder) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        if not self.image_files:
            self.get_logger().error(f"No images found in {image_folder}")
            return

        self.hf_token = os.getenv("HF_TOKEN")

        # Load OpenVLA 1B with 8-bit quantization
        self.get_logger().info("Loading OpenVLA 1B model (8-bit)...")
        self.processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            device_map="auto",
            load_in_8bit=True,
            trust_remote_code=True
        )
        self.get_logger().info("Model loaded.")

        # Example textual prompt for UR5e robot
        self.prompt = "UR5e robot: pick up the orange ball."

        # Start processing images
        self.process_images()

    def process_images(self):
        for img_path in self.image_files:
            self.get_logger().info(f"Processing image: {img_path}")
            image = PILImage.open(img_path).convert("RGB")

            # Prepare inputs for the model
            inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to(self.model.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model.generate(**inputs)
                text_output = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

            self.get_logger().info(f"Inference result: {text_output}")

def main(args=None):
    rclpy.init(args=args)

    # Replace with your folder path containing images
    image_folder = "/home/peter/uni/project_in_robotics/test_images"
    node = OpenVLANode(image_folder)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
