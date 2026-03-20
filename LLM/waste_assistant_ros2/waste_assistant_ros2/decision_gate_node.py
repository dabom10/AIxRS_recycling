import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .decision_logic import VisionHint, gate_and_decide


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
                'decision': {
                    'action': 'NO_MOVE_HUMAN',
                    'target_bin': 'unknown',
                    'p_fused': 0.0,
                    'preconditions': [],
                    'speak': '판단 중 오류가 발생했습니다. 관리자 확인이 필요합니다.',
                    'why': 'llm_error',
                },
                'raw_llm': payload,
            }
        else:
            vision = VisionHint.from_dict(payload['input'].get('vision_hint'))
            raw_llm = payload['raw_llm']
            decision = gate_and_decide(raw_llm, vision)
            out = {
                'ok': True,
                'request_id': payload.get('request_id'),
                'decision': decision,
                'raw_llm': raw_llm,
                'input': payload['input'],
            }
        output = String()
        output.data = json.dumps(out, ensure_ascii=False)
        self.publisher.publish(output)
        self.get_logger().info(f"Decision: {out['decision']['action']}")


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
