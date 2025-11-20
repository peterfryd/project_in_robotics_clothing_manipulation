#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from custom_interfaces_pkg.srv import GetLandmarks
from sensor_msgs.msg import Image
import threading
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from custom_interfaces_pkg.msg import Landmark
import cv2
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image as PILImage


class GetLandmarksNode(Node):
    def __init__(self):
        super().__init__('get_landmarks_node')

        # Image update lock
        self.lock = threading.Lock()
        self.image = None
        self.bridge = CvBridge()
        
        # Load Model
        model_name = "model.pth"
        self.pkg_path = get_package_share_directory('clothing_ai_pkg')
        model_path = os.path.join(self.pkg_path, 'data', model_name)
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model file not found at {model_path}")
            exit()
        
        self.num_landmarks = 25
        self.data_per_landmark = 3  # x and y, visibility
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
        model.fc = nn.Linear(model.fc.in_features, self.num_landmarks * self.data_per_landmark)
    
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            # Check if it's a checkpoint dict or just state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Remove 'backbone.' prefix if present
                if all(k.startswith('backbone.') for k in state_dict.keys()):
                    state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                self.get_logger().info(f"✅ Loaded model from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                model.load_state_dict(checkpoint)
                self.get_logger().info(f"✅ Loaded model from {model_path}")
        except FileNotFoundError:
            self.get_logger().error(f"❌ Model not found at {model_path}")
            exit()
            
        model.to(self.device)
        model.eval()
        return model
        
    def image_callback(self, msg):
        with self.lock:
            self.image = msg
    
    def get_landmarks(self, request, response):
        cv_image = None
        cv_crop = None
        ros_image = None
        diff = 1280-720
        left_crop = int(1/4*diff)
        right_crop = int(3/4*diff)

        with self.lock:
            if self.image is not None:
                ros_image = self.image
                cv_image = self.bridge.imgmsg_to_cv2(self.image, desired_encoding='bgr8')
                # Crop
                cv_crop = cv_image[:, left_crop:1280-right_crop]
                # Rotate image 90 degrees counter clockwise
                cv_crop = cv2.rotate(cv_crop, cv2.ROTATE_90_CLOCKWISE)
            else:
                self.get_logger().warning("No image available yet for landmark detection")
                return response  # return empty response gracefully

        # Run inference
        final_preds = self.run_inference(cv_crop)
        landmarks_list = [Landmark(x=float(pair[0]), y=float(pair[1])) for pair in final_preds]
        landmarks_list = [
            landmarks_list[14],
            landmarks_list[9],
            landmarks_list[8],
            landmarks_list[1],
            landmarks_list[5],
            landmarks_list[22],
            landmarks_list[21],
            landmarks_list[16]
        ]

        # Transform landmarks back to original image coordinates
        crop_height = 720
        landmarks_list_corrected = []
        for lm in landmarks_list:
            x_cropped = lm.y
            y_cropped = crop_height - lm.x
            
            # Add the left crop offset to get back to original image coordinates
            x_original = x_cropped + left_crop
            y_original = y_cropped
            
            landmarks_list_corrected.append(Landmark(x=float(x_original), y=float(y_original)))
        
        response.landmarks = landmarks_list_corrected
        response.image = ros_image
        
        # Annotate and save image
        self.annotate_image(cv_crop, landmarks_list, "landmarks")
        self.annotate_image(cv_image, landmarks_list_corrected, "landmarks_corrected")
        
        return response
        
    def run_inference(self, image):
        # Convert OpenCV image (BGR) to PIL RGB
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(image)
        else:
            raise RuntimeError("Invalid image type passed to run_inference")
        
        # Transform and create tensor
        tfms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        img_tensor = tfms(pil_image).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            preds = self.model(img_tensor).view(self.num_landmarks, self.data_per_landmark).cpu()

        # Extract only x, y coordinates (ignore visibility)
        preds_xy = preds[:, :2]

        # Scale model output [0-1] -> pixel coordinates
        final_preds = preds_xy.clone()
        final_preds[:, 0] *= pil_image.width
        final_preds[:, 1] *= pil_image.height

        landmarks = final_preds.tolist()
        return landmarks

    def annotate_image(self, image, landmarks, image_name):
        # Save to outermost directory (workspace root)
        workspace_root = '/home/'
        
        for lm in landmarks:  # lm is a Landmark object
            x = lm.x
            y = lm.y
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
            # Put text label coordiantes next to landmark
            cv2.putText(image, f"({int(x)},{int(y)})", (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            # Also put landmark index
            index = landmarks.index(lm)
            cv2.putText(image, f"{index}", (int(x)-10, int(y)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        save_path = os.path.join(workspace_root, image_name + '.png')
        cv2.imwrite(save_path, image)
        self.get_logger().info(f"Saved visualized result to {save_path}")



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
