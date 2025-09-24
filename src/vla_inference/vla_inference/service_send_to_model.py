from vla_inference.srv import ServiceSendToModel

import rclpy
from rclpy.node import Node


class MinimalService(Node):

    def __init__(self):
        super.__init__('Minimal_service')
        self.srv =  self.create_service()