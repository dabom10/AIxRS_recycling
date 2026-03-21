import os
import sys
import types

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import cv2
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32

# ── grasp_detection_node 경로 설정 ───────────────────────────────────────────
# __file__ 위치에서 위로 올라가며 CV/grasp_detection/grasp_detection 을 탐색
def _find_grasp_dir():
    target = os.path.join("CV", "grasp_detection", "grasp_detection")
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        candidate = os.path.join(current, target)
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    raise RuntimeError(
        f"grasp_detection 디렉터리를 찾을 수 없습니다. "
        f"CV/grasp_detection/grasp_detection 경로가 워크스페이스 내에 있는지 확인하세요."
    )

_GRASP_DIR = _find_grasp_dir()

_MODEL_DIR = os.path.abspath(os.path.join(_GRASP_DIR, "../../models"))

# ── grasp_detection_node import 준비 ─────────────────────────────────────────
# config를 직접 로드 후 AIxRS_recycling... 경로로 sys.modules에 등록
sys.path.insert(0, _GRASP_DIR)

# cv_bridge는 NumPy 1.x 기준 컴파일 → NumPy 2.x에서 _ARRAY_API 오류 발생
# detect.py는 cv_bridge를 직접 사용하지 않으므로 mock으로 우회
from unittest.mock import MagicMock
for _cv_mod in ['cv_bridge', 'cv_bridge.boost', 'cv_bridge.boost.cv_bridge_boost']:
    sys.modules.setdefault(_cv_mod, MagicMock())

import config as _gd_config

for _mod_name in [
    'AIxRS_recycling',
    'AIxRS_recycling.CV',
    'AIxRS_recycling.CV.grasp_detection',
    'AIxRS_recycling.CV.grasp_detection.grasp_detection',
]:
    sys.modules.setdefault(_mod_name, types.ModuleType(_mod_name))

sys.modules['AIxRS_recycling.CV.grasp_detection.grasp_detection.config'] = _gd_config

# get_package_share_directory mock (grasp_detection 패키지 미설치 대응)
from unittest.mock import patch
with patch('ament_index_python.packages.get_package_share_directory', return_value=_MODEL_DIR):
    import grasp_detection_node as _gdn

# ── 함수 및 설정 참조 ─────────────────────────────────────────────────────────
make_roi                    = _gdn.make_roi
crop_roi                    = _gdn.crop_roi
pick_best_detection         = _gdn.pick_best_detection
bbox_geometry               = _gdn.bbox_geometry
get_class_name              = _gdn.get_class_name
detect_yolo_boxes_from_path = _gdn.detect_yolo_boxes_from_path

ROI      = _gd_config.ROI
SELECT   = _gd_config.SELECT
YOLO_SEG = _gd_config.YOLO_SEG

class RecycleDetect(Node):
    def __init__(self):
        super().__init__("recycle_detect")

        # 모델 로드
        model_path = os.path.join(_MODEL_DIR, YOLO_SEG.model_path)
        self.get_logger().info(f"모델 로딩 중: {model_path}")
        self.model = YOLO(model_path)
        self.model.conf = YOLO_SEG.conf
        self.model.iou = YOLO_SEG.iou
        self.model.agnostic = YOLO_SEG.agnostic
        self.model.multi_label = YOLO_SEG.multi_label
        self.model.max_det = YOLO_SEG.max_det
        self.get_logger().info("모델 로딩 완료")

        # 임시 ROI 저장 디렉터리
        import tempfile
        self._tmp_dir = tempfile.mkdtemp(prefix="recycle_detect_")
        self._frame_idx = 0

        # 웹캠
        self.cap = cv2.VideoCapture(8)
        if not self.cap.isOpened():
            self.get_logger().error("웹캠을 열 수 없습니다.")
            raise RuntimeError("웹캠 오류")

        # 퍼블리셔
        self.result_pub   = self.create_publisher(Bool,    "/recycle1/grasp_result", 10)
        self.class_pub    = self.create_publisher(String,  "/recycle1/class_name",   10)
        self.cx_pub       = self.create_publisher(Float32, "/recycle1/center_x",     10)
        self.cy_pub       = self.create_publisher(Float32, "/recycle1/center_y",     10)
        self.depth_pub    = self.create_publisher(Float32, "/recycle1/depth",        10)

        # 타이머 (30fps)
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.get_logger().info("RecycleDetect 시작 | 토픽: class_name / center_x / center_y / depth")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("프레임 읽기 실패")
            return

        self._frame_idx += 1
        result, vis = self._process(frame, self._frame_idx)

        # 감지 성공 여부 퍼블리시
        msg = Bool()
        msg.data = bool(result.get("ok", False))
        self.result_pub.publish(msg)

        # 객체명 퍼블리시
        class_msg = String()
        class_msg.data = result.get("class", "")
        if class_msg.data:
            self.class_pub.publish(class_msg)

        # center_x, center_y, depth 퍼블리시
        cx, cy = result.get("centroid_uv", [0.0, 0.0])
        cx_msg = Float32(); cx_msg.data = float(cx)
        cy_msg = Float32(); cy_msg.data = float(cy)
        depth_msg = Float32(); depth_msg.data = result.get("depth", 0.0)
        self.cx_pub.publish(cx_msg)
        self.cy_pub.publish(cy_msg)
        self.depth_pub.publish(depth_msg)

        # 화면 출력
        cv2.imshow("RecycleDetect", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.get_logger().info("종료")
            self.cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()

    def _process(self, frame, frame_idx: int):
        roi = make_roi(frame, ROI.x1_ratio, ROI.y1_ratio, ROI.x2_ratio, ROI.y2_ratio)
        x1r, y1r, x2r, y2r = roi
        color_roi = crop_roi(frame, roi)

        # ROI를 임시 파일로 저장 후 추론 (grasp_detection_node 방식)
        roi_path = os.path.join(self._tmp_dir, f"roi_{frame_idx:06d}.jpg")
        cv2.imwrite(roi_path, color_roi)
        candidates = detect_yolo_boxes_from_path(
            model=self.model,
            image_path=roi_path,
            imgsz=YOLO_SEG.imgsz,
        )

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
