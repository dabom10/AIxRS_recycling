import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class DecisionGateNode(Node):
    def __init__(self):
        super().__init__('decision_gate_node')
        self.subscription = self.create_subscription(String, '/waste/raw_llm', self.on_raw_llm, 10)
        self.publisher = self.create_publisher(String, '/waste/decision', 10)
        self.get_logger().info('Decision gate ready.')

    def on_raw_llm(self, msg: String) -> None:
        payload = json.loads(msg.data)
        if not payload.get('ok'):
            out = {
                'ok': False,
                'request_id': payload.get('request_id'),
                'decision': {
                    'image_path': None,
                    'classification_possible': False,
                    'object_name': None,
                    'selected_bbox_index': None,
                    'selected_bbox_xyxy': None,
                    'selected_center_xy': None,
                    'all_bboxes': [],
                    'needs_human_action': True,
                    'human_action_reason': payload.get('error', 'llm_error'),
                    'robot_action': 'STOP',
                    'bin_type': None,
                    'answer': '판단 중 오류가 발생했습니다. 관리자 확인이 필요합니다.',
                },
                'raw_llm': payload,
            }
        else:
            raw_llm = payload['raw_llm']
            out = {
                'ok': True,
                'request_id': payload.get('request_id'),
                'decision': raw_llm,
                'input': payload.get('input', {}),
                'raw_llm': raw_llm,
            }
        output = String()
        output.data = json.dumps(out, ensure_ascii=False)
        self.publisher.publish(output)
        self.get_logger().info(f"Decision: {out['decision']['robot_action']}")


def main(args=None):
    rclpy.init(args=args)
    node = DecisionGateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
