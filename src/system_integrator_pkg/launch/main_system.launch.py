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
            executable='fold_point_to_point',
            name='fold_point_to_point',
            output='screen'
        ),
        Node(
            package='image_processing_pkg',
            executable='image_to_base',
            name='image_to_base',
            output='screen'
        ),
        Node(
            package='image_processing_pkg',
            executable='get_pick_and_place_point',
            name='get_pick_and_place_point',
            output='screen'
        )
    ])
