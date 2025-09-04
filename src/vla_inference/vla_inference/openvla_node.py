# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import String  # Replace with proper message type
# from transformers import AutoModelForVision2Seq, AutoProcessor
# import torch
# from PIL import Image
# import numpy as np
# import cv2

# class OpenVLANode(Node):
#     def __init__(self):
#         super().__init__('openvla_node')
#         self.get_logger().info("OpenVLA Node started.")

#         # Example subscriber (replace 'input_topic' with your actual topic)
#         self.subscription = self.create_subscription(
#             String,
#             'input_topic',
#             self.callback,
#             10
#         )

#         # Load RT-2X model
#         self.get_logger().info("Loading RT-2X model...")
#         self.processor = AutoProcessor.from_pretrained("openvla/rt-2x")
#         self.model = AutoModelForVision2Seq.from_pretrained("openvla/rt-2x").to("cuda" if torch.cuda.is_available() else "cpu")
#         self.get_logger().info("Model loaded.")

#     def callback(self, msg):
#         # Example: treat msg.data as path to an image
#         image_path = msg.data
#         self.get_logger().info(f"Received image path: {image_path}")

#         image = Image.open(image_path).convert("RGB")
#         inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)

#         with torch.no_grad():
#             outputs = self.model.generate(**inputs)
#             text_output = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

#         self.get_logger().info(f"Inference result: {text_output}")

# def main(args=None):
#     rclpy.init(args=args)
#     node = OpenVLANode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from PIL import Image
import numpy as np
import cv2

class OpenVLANode(Node):
    def __init__(self):
        super().__init__('openvla_node')
        self.get_logger().info("Hi! All imports loaded successfully.")

        # Example subscriber (not really used, just to show structure)
        self.subscription = self.create_subscription(
            String,
            'input_topic',
            self.callback,
            10
        )

    def callback(self, msg):
        self.get_logger().info(f"Received message: {msg.data}")

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
