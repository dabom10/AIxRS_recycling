import json
from typing import List, Dict, Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO

from .config import ROI, YOLO_SEG, TOPIC


class GraspDetectionNode(Node):
    def __init__(self) -> None:
        super().__init__("grasp_detection_node")

        self.bridge = CvBridge()
        self.model = YOLO(YOLO_SEG.model_path)

        self.image_sub = self.create_subscription(
            Image,
            TOPIC.image_topic,
            self.image_callback,
            10,
        )
        self.detection_pub = self.create_publisher(String, TOPIC.detection_topic, 10)
        self.debug_image_pub = self.create_publisher(Image, TOPIC.debug_image_topic, 10)

        self.get_logger().info(f"Loaded YOLO model: {YOLO_SEG.model_path}")
        self.get_logger().info(f"Subscribed image topic: {TOPIC.image_topic}")
        self.get_logger().info(f"Publishing detections to: {TOPIC.detection_topic}")

    @staticmethod
    def make_roi(image: np.ndarray, x1_ratio: float, y1_ratio: float, x2_ratio: float, y2_ratio: float):
        h, w = image.shape[:2]

        x1 = int(w * x1_ratio)
        y1 = int(h * y1_ratio)
        x2 = int(w * x2_ratio)
        y2 = int(h * y2_ratio)

        x1 = max(0, min(x1, w - 2))
        y1 = max(0, min(y1, h - 2))
        x2 = max(x1 + 1, min(x2, w - 1))
        y2 = max(y1 + 1, min(y2, h - 1))
        return x1, y1, x2, y2

    @staticmethod
    def crop_roi(img: np.ndarray, roi):
        x1, y1, x2, y2 = roi
        return img[y1:y2, x1:x2].copy()

    @staticmethod
    def get_class_name(model: YOLO, cls_id: int) -> str:
        names = model.names
        if isinstance(names, dict):
            return names.get(cls_id, str(cls_id))
        if isinstance(names, list) and 0 <= cls_id < len(names):
            return names[cls_id]
        return str(cls_id)

    def run_yolov8_seg(self, image: np.ndarray):
        results = self.model.predict(
            source=image,
            conf=YOLO_SEG.conf,
            iou=YOLO_SEG.iou,
            agnostic_nms=YOLO_SEG.agnostic_nms,
            max_det=YOLO_SEG.max_det,
            save=YOLO_SEG.save,
            verbose=YOLO_SEG.verbose,
        )
        return results[0]

    def extract_candidates_full_coords(self, result, roi) -> List[Dict[str, Any]]:
        x1r, y1r, x2r, y2r = roi
        boxes = result.boxes
        masks = result.masks

        detections: List[Dict[str, Any]] = []
        if boxes is None or len(boxes) == 0:
            return detections

        mask_polygons = masks.xy if (masks is not None and masks.xy is not None) else None

        roi_h = y2r - y1r
        roi_w = x2r - x1r
        target_cx_roi = roi_w / 2.0
        target_cy_roi = roi_h / 2.0

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            conf = float(boxes.conf[i].item())
            cls_id = int(boxes.cls[i].item())
            class_name = self.get_class_name(self.model, cls_id)

            bw = max(1, int(x2) - int(x1))
            bh = max(1, int(y2) - int(y1))
            area = bw * bh

            center_x_roi = (int(x1) + int(x2)) / 2.0
            center_y_roi = (int(y1) + int(y2)) / 2.0

            if mask_polygons is not None and i < len(mask_polygons):
                poly = mask_polygons[i]
                if poly is not None and len(poly) > 0:
                    center_x_roi = float(np.mean(poly[:, 0]))
                    center_y_roi = float(np.mean(poly[:, 1]))

            dist2 = float((center_x_roi - target_cx_roi) ** 2 + (center_y_roi - target_cy_roi) ** 2)

            detections.append(
                {
                    "idx": int(i),
                    "bbox_xyxy": [
                        int(x1 + x1r),
                        int(y1 + y1r),
                        int(x2 + x1r),
                        int(y2 + y1r),
                    ],
                    "conf": float(conf),
                    "cls_id": int(cls_id),
                    "class_name": str(class_name),
                    "area": int(area),
                    "center": [
                        float(center_x_roi + x1r),
                        float(center_y_roi + y1r),
                    ],
                    "dist2": float(dist2),
                }
            )

        return detections

    @staticmethod
    def draw_bboxes_on_full_image(img: np.ndarray, detections, roi) -> np.ndarray:
        vis = img.copy()
        rx1, ry1, rx2, ry2 = roi

        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
        cv2.putText(vis, "ROI", (rx1, max(ry1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        target_cx = int((rx1 + rx2) / 2)
        target_cy = int((ry1 + ry2) / 2)
        cv2.circle(vis, (target_cx, target_cy), 6, (255, 0, 0), -1)
        cv2.putText(vis, "target", (target_cx + 8, target_cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            cx, cy = det["center"]
            label = f"{det['class_name']} {det['conf']:.2f} A:{det['area']}"

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            cv2.putText(vis, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return vis

    def image_callback(self, msg: Image) -> None:
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Failed to convert image: {exc}")
            return

        roi = self.make_roi(img, ROI.x1_ratio, ROI.y1_ratio, ROI.x2_ratio, ROI.y2_ratio)
        roi_img = self.crop_roi(img, roi)

        result = self.run_yolov8_seg(roi_img)
        detections = self.extract_candidates_full_coords(result, roi)

        output = {
            "ok": len(detections) > 0,
            "roi_xyxy": [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])],
            "target_center_full": [float((roi[0] + roi[2]) / 2.0), float((roi[1] + roi[3]) / 2.0)],
            "detections": detections,
        }

        self.detection_pub.publish(String(data=json.dumps(output, ensure_ascii=False)))

        if TOPIC.publish_debug_image:
            debug_img = self.draw_bboxes_on_full_image(img, detections, roi)
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            debug_msg.header = msg.header
            self.debug_image_pub.publish(debug_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraspDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
