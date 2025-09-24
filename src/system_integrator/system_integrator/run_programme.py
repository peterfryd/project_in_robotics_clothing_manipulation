#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from system_integrator.srv import Programme
from vla_inference.srv import ServiceSendToModel


class RunProgramme(Node):
    def __init__(self):
        super().__init__('run_programme')
        self.bridge = CvBridge()
        self.latest_image = None

        # Subscribe to camera
        self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)

        # Create service
        self.srv = self.create_service(Programme, 'run_programme', self.run_programme)
        
        # Create client for the other service
        self.other_service_client = self.create_client(ServiceSendToModel, 'INSERT NAME HERE')

        # Wait until the other service is available
        while not self.other_service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /move_command service...')

    def image_callback(self, msg):
        self.latest_image = msg  # Store as ROS Image

    def run_programme(self, request, response):
        if self.latest_image is None:
            self.get_logger().warn("No image received yet.")
            response.success = False
        
        
         # --- CALL THE OTHER SERVICE ---
        other_request = MoveCommand.Request()
        other_request.prompt = request.prompt
        other_request.image = self.latest_image

        # Call synchronously (blocking) — you could also do async
        future = self.other_service_client.call_async(other_request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            result = future.result()
            response.delta_position = result.delta_position
            response.delta_gripper = result.delta_gripper
            response.success = True
            self.get_logger().info(f"Received response from /move_command: {result}")
        else:
            self.get_logger().error("Failed to call /move_command")
            response.success = False
        
        response.success = True 
        
        return response


def main(args=None):
    rclpy.init(args=args)
    node = RunProgramme()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
