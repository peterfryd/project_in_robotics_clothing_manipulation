#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime


class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        
        # Declare parameters
        self.declare_parameter('topic_name', '/camera/camera/color/image_raw')
        self.declare_parameter('save_directory', './saved_images')
        
        # Get parameters
        self.topic_name = self.get_parameter('topic_name').value
        self.save_directory = self.get_parameter('save_directory').value
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_directory, exist_ok=True)
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Create subscription
        self.subscription = self.create_subscription(
            Image,
            self.topic_name,
            self.image_callback,
            10
        )
        
        self.get_logger().info(f'Image saver node started')
        self.get_logger().info(f'Subscribing to topic: {self.topic_name}')
        self.get_logger().info(f'Saving images to: {self.save_directory}')
        self.get_logger().info('Will save one image and exit')
    
    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f'image_{timestamp}.png'
            filepath = os.path.join(self.save_directory, filename)
            
            # Save image
            cv2.imwrite(filepath, cv_image)
            
            self.get_logger().info(f'Saved image: {filename}')
            self.get_logger().info('Image saved successfully. Shutting down...')
            
            # Shutdown after saving one image
            raise KeyboardInterrupt
            
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.get_logger().error(f'Failed to save image: {str(e)}')
            raise KeyboardInterrupt


def main(args=None):
    rclpy.init(args=args)
    
    image_saver = ImageSaver()
    
    try:
        rclpy.spin(image_saver)
    except KeyboardInterrupt:
        pass
    finally:
        image_saver.get_logger().info('Shutting down image saver')
        image_saver.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
