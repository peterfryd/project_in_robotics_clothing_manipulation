#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import threading
from sensor_msgs.msg import Image
from system_integrator.srv import Programme
from vla_inference.srv import ServiceSendToModel
from robot_controller_pkg.srv import RobotCmd

class RunProgramme(Node):
    def __init__(self,):
        super().__init__('run_programme')
        self.latest_image = None
        self.image_lock = threading.Lock()

        # Subscribe to camera
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)

        # Create service
        self.srv = self.create_service(Programme, 'service_run_programme', self.run_programme)
        
        # Create client for service_send_to_model
        self.ai_service = self.create_client(ServiceSendToModel, 'service_send_to_model')
        
        # Create client for service_send_to_model
        self.robot_service = self.create_client(RobotCmd, 'robot_cmd_service')

        # Wait until the other service is available
        while not self.ai_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service_send_to_model...')
            
             # Wait until the other service is available
        while not self.robot_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for robot_cmd_service...')

    def image_callback(self, msg):
        with self.image_lock:
            self.latest_image = msg  # Store as ROS Image
    
    def run_programme(self, request, response):
        img = None
        with self.image_lock:
            if self.latest_image is None:
                self.get_logger().warn("No image received yet.")
                response.success = False
                return response
            else: 
                img = self.latest_image
        
        
        #  # --- CALL THE AI SERVICE ---
        ai_request = ServiceSendToModel.Request()
        ai_request.prompt = request.prompt
        
        ai_request.image = img

        # Call synchronously (blocking) — you could also do async
        future = self.ai_service.call_async(ai_request)
        rclpy.spin_until_future_complete(self, future)
        
        delta_position = None
        delta_orientation = None
        delta_gripper = None
        
        if future.result() is not None:
            result = future.result()
            delta_position = result.delta_position
            delta_orientation = result.delta_orientation
            delta_gripper = result.delta_gripper
            self.get_logger().info(f"Received response from service_send_to_model: {result}")
        else:
            self.get_logger().error("Failed to call service_send_to_model")
            response.success = False
            return response
        
        
        #  # --- CALL THE ROBOT SERVICE ---
        robot_request = RobotCmd.Request()
        
        robot_request.delta_position = delta_position
        robot_request.delta_orientation = delta_orientation
        robot_request.delta_gripper = delta_gripper

        # Call synchronously (blocking) — you could also do async
        future = self.robot_service.call_async(robot_request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            result = future.result()
            response.success = result.success
            self.get_logger().info(f"Received response from robot_cmd_service: {result}")
        else:
            self.get_logger().error("Failed to call robot_cmd_service")
            response.success = False
            return response        
    
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
