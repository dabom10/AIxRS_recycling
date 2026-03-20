import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import json
import math
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import yolov5

import rclpy
from rclpy.node import Node
from AIxRS_recycling.CV.grasp_detection.grasp_detection.config import ROI, SELECT, YOLO_SEG
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import os
from ament_index_python.packages import get_package_share_directory

pkg_path = get_package_share_directory("grasp_detection")
full_model_path = os.path.join(pkg_path, "models", YOLO_SEG.model_path)

def make_roi(image, x1_ratio, y1_ratio, x2_ratio, y2_ratio):
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


def crop_roi(img, roi):
    x1, y1, x2, y2 = roi
    return img[y1:y2, x1:x2].copy()


def detect_yolo_boxes_from_path(model, image_path, imgsz=416):
    results = model(image_path, size=imgsz)
    predictions = results.pred[0]

    out = []
    if predictions is None or len(predictions) == 0:
        return out

    for pred in predictions:
        x1, y1, x2, y2, conf, cls = pred.tolist()
        out.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "conf": float(conf),
            "cls": int(cls),
        })
    return out


def get_class_name(model, cls_id):
    names = model.names
    if isinstance(names, dict):
        return names.get(cls_id, str(cls_id))
    elif isinstance(names, list) and 0 <= cls_id < len(names):
        return names[cls_id]
    return str(cls_id)


def draw_all_candidates_on_roi(roi_img, candidates, model):
    vis = roi_img.copy()

    for item in candidates:
        x1, y1, x2, y2 = item["bbox"]
        conf = item["conf"]
        cls_id = item["cls"]
        class_name = get_class_name(model, cls_id)
        label = f"{class_name} {conf:.2f}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return vis


def pick_best_detection(candidates, cfg, roi_shape):
    if not candidates:
        return None

    h, w = roi_shape[:2]
    cx0 = w / 2.0
    cy0 = h * 0.78
    diag = math.hypot(w, h)

    best = None
    best_score = -1e18

    for item in candidates:
        x1, y1, x2, y2 = item["bbox"]
        conf = item["conf"]

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area = bw * bh

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if area < cfg.min_area_px:
            continue

        if cy < h * cfg.min_cy_ratio:
            continue

        dist = math.hypot(cx - cx0, cy - cy0) / max(diag, 1e-6)
        bottom_bias = cy / float(h)

        score = (
            cfg.center_weight * (1.0 - dist)
            + cfg.area_weight * (area / float(w * h))
            + cfg.bottom_bonus * bottom_bias
            + 0.8 * conf
        )

        margin = 5
        if x1 <= margin or y1 <= margin or x2 >= (w - margin) or y2 >= (h - margin):
            score -= cfg.edge_penalty

        if score > best_score:
            best_score = score
            best = item

    if best is None and candidates:
        best = max(candidates, key=lambda x: x["conf"])

    return best


def bbox_geometry(bbox):
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    if bh >= bw:
        major_axis_angle_deg = 90.0
        grasp_angle_deg = 0.0
        grasp_line_xyxy = [
            float(x1), float(cy),
            float(x2), float(cy),
        ]
        grasp_width_px = float(bw)
    else:
        major_axis_angle_deg = 0.0
        grasp_angle_deg = 90.0
        grasp_line_xyxy = [
            float(cx), float(y1),
            float(cx), float(y2),
        ]
        grasp_width_px = float(bh)

    return {
        "centroid": (cx, cy),
        "object_width_px": float(bw),
        "object_height_px": float(bh),
        "major_axis_angle_deg": float(major_axis_angle_deg),
        "grasp_angle_deg": float(grasp_angle_deg),
        "grasp_line_xyxy": grasp_line_xyxy,
        "grasp_width_px": float(grasp_width_px),
    }


def draw_selected_debug(color, roi, yolo_bbox, centroid, label_text, major_angle, grasp_angle, grasp_line):
    vis = color.copy()
    x1r, y1r, x2r, y2r = roi

    # ROI
    cv2.rectangle(vis, (x1r, y1r), (x2r, y2r), (100, 100, 100), 2)

    # selected bbox
    x1, y1, x2, y2 = yolo_bbox
    x1 += x1r
    y1 += y1r
    x2 += x1r
    y2 += y1r

    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        vis,
        label_text,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    # centroid
    cx, cy = centroid
    cx = int(round(cx + x1r))
    cy = int(round(cy + y1r))
    cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)

    # grasp line
    gx1, gy1, gx2, gy2 = grasp_line
    p1 = (int(round(gx1 + x1r)), int(round(gy1 + y1r)))
    p2 = (int(round(gx2 + x1r)), int(round(gy2 + y1r)))
    cv2.line(vis, p1, p2, (255, 0, 0), 3)

    txt1 = f"major={major_angle:.1f}"
    txt2 = f"grasp={grasp_angle:.1f}"
    cv2.putText(vis, txt1, (x1r + 10, y1r + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, txt2, (x1r + 10, y1r + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return vis


class GraspDetectionNode(Node):
    def __init__(self):
        super().__init__("grasp_detection_node")

        self.bridge = CvBridge()

        # parameter
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("result_topic", "/grasp_detection/result")
        self.declare_parameter("debug_topic", "/grasp_detection/debug_image")
        self.declare_parameter("debug_roi_dir", "/tmp/grasp_detection_roi")
        self.declare_parameter("save_debug_roi", True)

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        result_topic = self.get_parameter("result_topic").get_parameter_value().string_value
        debug_topic = self.get_parameter("debug_topic").get_parameter_value().string_value
        self.debug_roi_dir = self.get_parameter("debug_roi_dir").get_parameter_value().string_value
        self.save_debug_roi = self.get_parameter("save_debug_roi").get_parameter_value().bool_value

        os.makedirs(self.debug_roi_dir, exist_ok=True)

        self.get_logger().info(f"Loading model: {YOLO_SEG.model_path}")
        self.model = yolov5.load(full_model_path)
        self.model.conf = YOLO_SEG.conf
        self.model.iou = YOLO_SEG.iou
        self.model.agnostic = YOLO_SEG.agnostic
        self.model.multi_label = YOLO_SEG.multi_label
        self.model.max_det = YOLO_SEG.max_det

        self.sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )

        self.result_pub = self.create_publisher(String, result_topic, 10)
        self.debug_pub = self.create_publisher(Image, debug_topic, 10)

        self.frame_idx = 0

        self.get_logger().info("GraspDetectionNode started")
        self.get_logger().info(f"subscribe: {image_topic}")
        self.get_logger().info(f"publish result: {result_topic}")
        self.get_logger().info(f"publish debug image: {debug_topic}")

    def publish_json(self, data: Dict[str, Any]):
        msg = String()
        msg.data = json.dumps(data, ensure_ascii=False)
        self.result_pub.publish(msg)

    def publish_debug_image(self, cv_img):
        try:
            msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
            self.debug_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish debug image: {e}")

    def image_callback(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge convert failed: {e}")
            return

        self.frame_idx += 1

        try:
            result, debug_img = self.process_image(img, self.frame_idx)
            self.publish_json(result)

            if debug_img is not None:
                self.publish_debug_image(debug_img)

        except Exception as e:
            self.get_logger().error(f"process_image failed: {e}")
            error_result = {
                "ok": False,
                "reason": "exception",
                "detail": str(e),
                "frame_id": self.frame_idx,
            }
            self.publish_json(error_result)

    def process_image(self, img, frame_idx: int):
        roi = make_roi(img, ROI.x1_ratio, ROI.y1_ratio, ROI.x2_ratio, ROI.y2_ratio)
        color_roi = crop_roi(img, roi)

        roi_path = os.path.join(self.debug_roi_dir, f"roi_{frame_idx:06d}.jpg")

        if self.save_debug_roi:
            cv2.imwrite(roi_path, color_roi)
            candidates = detect_yolo_boxes_from_path(
                model=self.model,
                image_path=roi_path,
                imgsz=YOLO_SEG.imgsz
            )
        else:

            cv2.imwrite(roi_path, color_roi)
            candidates = detect_yolo_boxes_from_path(
                model=self.model,
                image_path=roi_path,
                imgsz=YOLO_SEG.imgsz
            )

        roi_debug = draw_all_candidates_on_roi(color_roi, candidates, self.model)

        if not candidates:
            result = {
                "ok": False,
                "reason": "no_detection",
                "frame_id": frame_idx,
                "roi_xyxy": [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])],
                "all_detections": [],
            }
            return result, roi_debug

        selected_item = pick_best_detection(candidates, SELECT, color_roi.shape)
        if selected_item is None:
            result = {
                "ok": False,
                "reason": "no_selected_detection",
                "frame_id": frame_idx,
                "roi_xyxy": [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])],
                "all_detections": candidates,
            }
            return result, roi_debug

        yolo_bbox = selected_item["bbox"]
        geom = bbox_geometry(yolo_bbox)
        centroid = geom["centroid"]

        class_name = get_class_name(self.model, selected_item["cls"])
        label_text = f"{class_name} {selected_item['conf']:.2f}"

        debug_img = draw_selected_debug(
            color=img,
            roi=roi,
            yolo_bbox=yolo_bbox,
            centroid=centroid,
            label_text=label_text,
            major_angle=geom["major_axis_angle_deg"],
            grasp_angle=geom["grasp_angle_deg"],
            grasp_line=geom["grasp_line_xyxy"],
        )

        cx, cy = centroid
        x1r, y1r, _, _ = roi
        gx1, gy1, gx2, gy2 = geom["grasp_line_xyxy"]

        result = {
            "ok": True,
            "frame_id": frame_idx,
            "roi_xyxy": [
                int(roi[0]),
                int(roi[1]),
                int(roi[2]),
                int(roi[3]),
            ],
            "yolo_bbox_xyxy": [
                int(yolo_bbox[0] + x1r),
                int(yolo_bbox[1] + y1r),
                int(yolo_bbox[2] + x1r),
                int(yolo_bbox[3] + y1r),
            ],
            "bbox_xyxy": [
                int(yolo_bbox[0] + x1r),
                int(yolo_bbox[1] + y1r),
                int(yolo_bbox[2] + x1r),
                int(yolo_bbox[3] + y1r),
            ],
            "centroid_uv": [
                float(cx + x1r),
                float(cy + y1r),
            ],
            "object_width_px": float(geom["object_width_px"]),
            "object_height_px": float(geom["object_height_px"]),
            "major_axis_angle_deg": float(geom["major_axis_angle_deg"]),
            "grasp_angle_deg": float(geom["grasp_angle_deg"]),
            "grasp_width_px": float(geom["grasp_width_px"]),
            "grasp_line_xyxy": [
                float(gx1 + x1r),
                float(gy1 + y1r),
                float(gx2 + x1r),
                float(gy2 + y1r),
            ],
            "selected_conf": float(selected_item["conf"]),
            "selected_class": int(selected_item["cls"]),
            "selected_class_name": class_name,
            "all_detections": candidates,
        }

        return result, debug_img


def main(args=None):
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