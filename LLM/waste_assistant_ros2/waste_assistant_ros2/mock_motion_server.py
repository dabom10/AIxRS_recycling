import time

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node

from waste_sort_interfaces.action import SortObject


class MockMotionServer(Node):
    def __init__(self):
        super().__init__('mock_motion_server')
        self.server = ActionServer(self, SortObject, '/sort_object', execute_callback=self.execute_callback)
        self.get_logger().info('Mock motion action server ready.')

    def execute_callback(self, goal_handle):
        feedback = SortObject.Feedback()
        stages = ['pick', 'move', 'drop']
        for idx, stage in enumerate(stages, start=1):
            feedback.progress = idx / len(stages)
            feedback.stage = stage
            goal_handle.publish_feedback(feedback)
            time.sleep(0.3)
        goal_handle.succeed()
        result = SortObject.Result()
        result.success = True
        result.message = f'{goal_handle.request.target_bin} bin에 배치 완료'
        return result


def main(args=None):
    rclpy.init(args=args)
    node = MockMotionServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
