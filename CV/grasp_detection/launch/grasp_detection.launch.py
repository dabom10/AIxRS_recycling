from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="grasp_detection",
            executable="grasp_detection_node",
            name="grasp_detection_node",
            output="screen",
            parameters=[
                {
                    "image_topic": "/camera/color/image_raw",
                    "result_topic": "/grasp_detection/result",
                    "debug_topic": "/grasp_detection/debug_image",
                    "debug_roi_dir": "/tmp/grasp_detection_roi",
                    "save_debug_roi": True,
                }
            ]
        )
    ])