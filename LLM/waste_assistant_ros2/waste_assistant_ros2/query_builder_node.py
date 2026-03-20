import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class QueryBuilderNode(Node):
    def __init__(self):
        super().__init__('query_builder_node')
        self.publisher = self.create_publisher(String, '/waste/query', 10)
        self.declare_parameter('image_path', '')
        self.declare_parameter('audio_path', '')
        self.declare_parameter('question', '이거 어디다 버려?')
        self.declare_parameter('vision_hint_json', '{}')
        self.timer = self.create_timer(1.0, self.publish_once)
        self.sent = False

    def publish_once(self):
        if self.sent:
            return
        image_path = self.get_parameter('image_path').value
        audio_path = self.get_parameter('audio_path').value
        question = self.get_parameter('question').value
        vision_hint_json = self.get_parameter('vision_hint_json').value

        payload = {
            'request_id': 'manual-query-0001',
            'image_path': image_path,
            'audio_path': audio_path if audio_path else None,
            'question': question,
            'vision_hint': json.loads(vision_hint_json),
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.publisher.publish(msg)
        self.get_logger().info(f'Published query for {Path(image_path).name}')
        self.sent = True


def main(args=None):
    rclpy.init(args=args)
    node = QueryBuilderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
