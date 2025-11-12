#!/usr/bin/env python3
from pyexpat import model
from typing import final
import rclpy
from rclpy.node import Node
import numpy as np
from custom_interfaces_pkg.srv import GetLandmarks
from sensor_msgs.msg import Image
import threading
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
import cv2
import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class GetLandmarksNode(Node):
    def __init__(self):
        super().__init__('get_landmarks_node')

        # Image update lock
        self.lock = threading.Lock()
        self.image = None
        self.bridge = CvBridge()
        
        # Load Model
        model_name = "head.pth"
        pkg_path = get_package_share_directory('clothing_ai_pkg')
        model_path = os.path.join(pkg_path, 'data', model_name)
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model file not found at {model_path}")
            exit()
        
        self.num_landmarks = 8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.image_size = 224
        self.model = self.load_model(model_path)
        
        # ROS2 Subscribers and Services
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

        self.get_logger().info("get_landmarks node ready and providing /get_landmarks_srv.")

    def load_model(self, model_path):
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, self.num_landmarks * 3)
    
        try:
            model_path.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            print(f"❌ Error: Model not found at {model_path}")
            exit()
            
        model.to(self.device)
        model.eval()
        return model
        
    def image_callback(self, msg):
        with self.lock:
            self.image = msg
    
    def get_landmarks(self, request, response):
        cv_image = None
        ros_image = None
        with self.lock:
            if self.image is not None:
                ros_image = self.image
                cv_image = self.bridge.imgmsg_to_cv2(self.image, desired_encoding='bgr8')
            else:
                self.get_logger().error("No Image available to find landmarks")
        cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        response.landmarks = self.run_inference(cv_image)
        response.image = ros_image
        return response
        
    def run_inference(self, image):
        # --- PREDICT ---
        tfms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        img_tensor = tfms(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Model outputs [0.0 - 1.0]
            preds = model(img_tensor).view(self.num_landmarks, 3).cpu()

        # Scale Model Output [0-1] -> Pixels
        final_preds = preds.clone()
        final_preds[:, 0] *= image.shape[1]
        final_preds[:, 1] *= image.shape[0]

        landmarks = final_preds.tolist()

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
