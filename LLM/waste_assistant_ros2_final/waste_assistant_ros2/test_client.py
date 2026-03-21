import json
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TestClient(Node):
    def __init__(self, image_path: str, audio_path: str, detections: list):
        super().__init__('test_client')
        self.publisher = self.create_publisher(String, '/waste/query', 10)
        self.timer = self.create_timer(1.0, self.publish_once)
        self.sent = False
        self.payload = {
            'request_id': 'test-req-0001',
            'image_path': image_path,
            'audio_path': audio_path,
            'detections': detections,
        }

    def publish_once(self):
        if self.sent:
            return
        msg = String()
        msg.data = json.dumps(self.payload, ensure_ascii=False)
        self.publisher.publish(msg)
        self.get_logger().info('Test query published.')
        self.sent = True


def main(args=None):
    args = args if args is not None else sys.argv

    image_path = args[1] if len(args) > 1 else '/tmp/plastic.jpg'
    audio_path = args[2] if len(args) > 2 else '/tmp/question.wav'

    detections = [
        {
            'idx': 0,
            'bbox_xyxy': [257, 170, 310, 223],
            'center': [283.5, 196.5],
            'class_name': 'plastic',
            'conf': 0.91,
        }
    ]

    rclpy.init(args=args)
    node = TestClient(image_path, audio_path, detections)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
