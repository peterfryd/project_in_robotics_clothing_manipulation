from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    
    camera_launch = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch',
        'rs_launch.py'
    )
    
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(camera_launch)
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
            package='clothing_ai_pkg',
            executable='get_landmarks_sift_cor',
            name='get_landmarks_sift_cor',
            output='screen'
        ),
        Node(
            package='image_processing_pkg',
            executable='get_pick_and_place_point',
            name='get_pick_and_place_point',
            output='screen'
        )
    ])
