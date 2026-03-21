from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='waste_assistant_ros2', executable='gemini_reasoner_node', output='screen'),
        Node(package='waste_assistant_ros2', executable='decision_gate_node', output='screen'),
        Node(package='waste_assistant_ros2', executable='sort_executor_node', output='screen'),
        Node(package='waste_assistant_ros2', executable='mock_motion_server', output='screen'),
        Node(package='waste_assistant_ros2', executable='hri_node', output='screen'),
    ])
