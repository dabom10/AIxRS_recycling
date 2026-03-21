import json
import traceback

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .gemini_service import GeminiWasteAssistant


class GeminiReasonerNode(Node):
    def __init__(self):
        super().__init__('gemini_reasoner_node')
        self.assistant = GeminiWasteAssistant()
        self.subscription = self.create_subscription(String, '/waste/query', self.on_query, 10)
        self.publisher = self.create_publisher(String, '/waste/raw_llm', 10)
        self.get_logger().info('Gemini reasoner ready.')

    def on_query(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            decision = self.assistant.ask(
                image_path=payload['image_path'],
                audio_path=payload.get('audio_path'),
                vision_hint=payload.get('vision_hint'),
            )
            out = {
                'ok': True,
                'request_id': payload.get('request_id'),
                'input': payload,
                'raw_llm': decision.model_dump(),
            }
        except Exception as exc:
            out = {
                'ok': False,
                'error': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc(),
            }
            self.get_logger().error(out['error'])
        output = String()
        output.data = json.dumps(out, ensure_ascii=False)
        self.publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = GeminiReasonerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
