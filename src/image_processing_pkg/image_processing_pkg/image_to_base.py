import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.service import Service

class ImageToBaseNode(Node):
    def __init__(self):
        super().__init__('image_to_base')
        self.get_logger().info("Node started")

        self.latest_depth_image = None

        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )

    def depth_callback(self, msg: Image):
        self.latest_depth_image = msg
        self.get_logger().info(f"Received depth image: {msg.width}x{msg.height}, encoding: {msg.encoding}")


def main(args=None):
    rclpy.init(args=args)
    node = ImageToBaseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
