import os
import sys
import json

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import cv2
import yolov5

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

# test_image_runner 경로 추가
_CV_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../CV/local_Test_grasp_detection")
)
sys.path.insert(0, _CV_DIR)

from test_image_runner import (
    make_roi,
    crop_roi,
    pick_best_detection,
    bbox_geometry,
    get_class_name,
)
from config import ROI, SELECT, YOLO_SEG


class RecycleDetect(Node):
    def __init__(self):
        super().__init__("recycle_detect")

        # 모델 로드
        model_path = os.path.abspath(os.path.join(_CV_DIR, YOLO_SEG.model_path))
        self.get_logger().info(f"모델 로딩 중: {model_path}")
        self.model = yolov5.load(model_path)
        self.model.conf = YOLO_SEG.conf
        self.model.iou = 0.45
        self.model.agnostic = False
        self.model.multi_label = True
        self.model.max_det = 1000
        self.get_logger().info("모델 로딩 완료")

        # 웹캠
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("웹캠을 열 수 없습니다.")
            raise RuntimeError("웹캠 오류")

        # 퍼블리셔
        self.result_pub = self.create_publisher(Bool,   "/recycle1/grasp_result", 10)
        self.class_pub  = self.create_publisher(String, "/recycle1/class_name", 10)

        # 타이머 (30fps)
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.get_logger().info("RecycleDetect 시작 | 토픽: /recycle1/class_name")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("프레임 읽기 실패")
            return

        result, vis = self._process(frame)

        # 감지 성공 여부 퍼블리시
        msg = Bool()
        msg.data = bool(result.get("ok", False))
        self.result_pub.publish(msg)

        # 객체명만 퍼블리시
        class_msg = String()
        class_msg.data = result.get("class", "")
        if class_msg.data:
            self.class_pub.publish(class_msg)

        # 화면 출력
        cv2.imshow("RecycleDetect", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.get_logger().info("종료")
            self.cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()

    def _process(self, frame):
        roi = make_roi(frame, ROI.x1_ratio, ROI.y1_ratio, ROI.x2_ratio, ROI.y2_ratio)
        x1r, y1r, x2r, y2r = roi
        color_roi = crop_roi(frame, roi)

        # numpy 배열 직접 추론
        results = self.model(color_roi, size=416)
        predictions = results.pred[0]

        candidates = []
        if predictions is not None and len(predictions) > 0:
            for pred in predictions:
                bx1, by1, bx2, by2, conf, cls = pred.tolist()
                candidates.append({
                    "bbox": [int(bx1), int(by1), int(bx2), int(by2)],
                    "conf": float(conf),
                    "cls":  int(cls),
                })

        vis = frame.copy()
        cv2.rectangle(vis, (x1r, y1r), (x2r, y2r), (100, 100, 100), 2)

        if not candidates:
            cv2.putText(vis, "No detection", (x1r + 10, y1r + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return {"ok": False, "reason": "no_detection"}, vis

        # 전체 후보 표시
        for item in candidates:
            bx1, by1, bx2, by2 = item["bbox"]
            cv2.rectangle(vis,
                          (bx1 + x1r, by1 + y1r),
                          (bx2 + x1r, by2 + y1r),
                          (0, 200, 0), 1)

        # 최적 객체 선택
        selected = pick_best_detection(candidates, SELECT, color_roi.shape)
        if selected is None:
            return {"ok": False, "reason": "no_selected"}, vis

        geom       = bbox_geometry(selected["bbox"])
        class_name = get_class_name(self.model, selected["cls"])
        label      = f"{class_name} {selected['conf']:.2f}"

        bx1, by1, bx2, by2 = selected["bbox"]
        cv2.rectangle(vis,
                      (bx1 + x1r, by1 + y1r),
                      (bx2 + x1r, by2 + y1r),
                      (0, 255, 0), 2)
        cv2.putText(vis, label,
                    (bx1 + x1r, max(by1 + y1r - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cx, cy = geom["centroid"]
        cv2.circle(vis, (int(cx + x1r), int(cy + y1r)), 6, (0, 0, 255), -1)

        gx1, gy1, gx2, gy2 = geom["grasp_line_xyxy"]
        cv2.line(vis,
                 (int(gx1 + x1r), int(gy1 + y1r)),
                 (int(gx2 + x1r), int(gy2 + y1r)),
                 (255, 0, 0), 3)

        cv2.putText(vis, f"grasp={geom['grasp_angle_deg']:.1f}deg",
                    (x1r + 10, y1r + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        result = {
            "ok":             True,
            "class":          class_name,
            "conf":           float(selected["conf"]),
            "centroid_uv":    [float(cx + x1r), float(cy + y1r)],
            "grasp_angle_deg": float(geom["grasp_angle_deg"]),
            "grasp_width_px": float(geom["grasp_width_px"]),
        }
        return result, vis

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RecycleDetect()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
