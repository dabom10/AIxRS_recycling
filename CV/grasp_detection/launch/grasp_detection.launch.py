from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='grasp_detection',
            executable='grasp_detection_node',
            name='grasp_detection_node',
            output='screen',
        )
    ])
