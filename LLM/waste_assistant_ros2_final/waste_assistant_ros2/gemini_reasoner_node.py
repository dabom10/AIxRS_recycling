import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .gemini_service import GeminiWasteAssistant


class GeminiReasonerNode(Node):
    def __init__(self):
        super().__init__('gemini_reasoner_node')
        self.subscription = self.create_subscription(String, '/waste/query', self.on_query, 10)
        self.publisher = self.create_publisher(String, '/waste/raw_llm', 10)
        self.assistant = GeminiWasteAssistant()
        self.get_logger().info('Gemini reasoner ready.')

    def on_query(self, msg: String) -> None:
        payload = json.loads(msg.data)
        request_id = payload.get('request_id', '')
        image_path = payload.get('image_path')
        audio_path = payload.get('audio_path')
        detections = payload.get('detections', [])

        try:
            result = self.assistant.ask(
                image_path=image_path,
                audio_path=audio_path,
                detections=detections,
            )
            out = {
                'ok': True,
                'request_id': request_id,
                'input': {
                    'image_path': image_path,
                    'audio_path': audio_path,
                    'detections': detections,
                },
                'raw_llm': result,
            }
        except Exception as exc:
            out = {
                'ok': False,
                'request_id': request_id,
                'error': str(exc),
                'input': {
                    'image_path': image_path,
                    'audio_path': audio_path,
                    'detections': detections,
                },
            }
            self.get_logger().error(f'Gemini 호출 실패: {exc}')

        message = String()
        message.data = json.dumps(out, ensure_ascii=False)
        self.publisher.publish(message)


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
