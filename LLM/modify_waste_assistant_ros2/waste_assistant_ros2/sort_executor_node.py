import json

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String

from waste_sort_interfaces.action import SortObject


class SortExecutorNode(Node):
    def __init__(self):
        super().__init__('sort_executor_node')
        self.subscription = self.create_subscription(String, '/waste/decision', self.on_decision, 10)
        self.publisher = self.create_publisher(String, '/waste/execution_result', 10)
        self.action_client = ActionClient(self, SortObject, '/sort_object')
        self._pending_request_id = None
        self.get_logger().info('Sort executor ready.')

    def on_decision(self, msg: String) -> None:
        payload = json.loads(msg.data)
        decision = payload['decision']
        action = decision['action']
        if not action.startswith('MOVE_'):
            out = String()
            out.data = json.dumps({
                'request_id': payload.get('request_id'),
                'executed': False,
                'decision': decision,
                'message': decision['speak'],
            }, ensure_ascii=False)
            self.publisher.publish(out)
            return

        goal = SortObject.Goal()
        goal.request_id = payload.get('request_id', '')
        goal.target_bin = decision['target_bin']
        goal.object_pose_frame = 'camera_link'
        goal.x = 0.4
        goal.y = 0.0
        goal.z = 0.2
        goal.qx = 0.0
        goal.qy = 0.0
        goal.qz = 0.0
        goal.qw = 1.0
        goal.grasp_profile = 'top_grasp'
        self._pending_request_id = goal.request_id

        self.action_client.wait_for_server()
        send_future = self.action_client.send_goal_async(goal, feedback_callback=self.feedback_callback)
        send_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(f'progress={fb.progress:.2f}, stage={fb.stage}')

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning('Goal rejected')
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        out = String()
        out.data = json.dumps({
            'request_id': self._pending_request_id,
            'executed': result.success,
            'message': result.message,
        }, ensure_ascii=False)
        self.publisher.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = SortExecutorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
