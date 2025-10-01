from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='realsense_camera',
            output='screen'
        ),
        Node(
            package='robot_controller_pkg',
            executable='arm_controller',
            name='arm_controller',
            output='screen'
        ),
        Node(
            package='vla_pkg',
            executable='inference',
            name='vla_inference',
            output='screen'
        ),
    ])
