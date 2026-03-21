import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class HRINode(Node):
    def __init__(self):
        super().__init__('hri_node')
        self.create_subscription(String, '/waste/decision', self.on_decision, 10)
        self.create_subscription(String, '/waste/execution_result', self.on_result, 10)
        self.get_logger().info('HRI node ready.')

    def on_decision(self, msg: String) -> None:
        payload = json.loads(msg.data)
        decision = payload['decision']
        self.get_logger().info(
            f"[HRI] action={decision.get('robot_action')} | obj={decision.get('object_name')} | "
            f"pixel={decision.get('selected_center_xy')} | msg={decision.get('answer')}"
        )

    def on_result(self, msg: String) -> None:
        payload = json.loads(msg.data)
        self.get_logger().info(
            f"[EXECUTION] executed={payload['executed']} | pixel={payload.get('selected_center_xy')} | {payload['message']}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = HRINode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
