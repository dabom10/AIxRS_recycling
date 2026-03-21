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
# _GRASP_DIR을 sys.path에 추가해 config를 직접 로드하고,
# _GRASP_DIR의 부모 디렉터리도 추가해 grasp_detection 패키지로 인식시킴
# → grasp_detection_node.py의 `from .config import ...` 상대 임포트가 정상 동작
_GRASP_PARENT_DIR = os.path.dirname(_GRASP_DIR)
sys.path.insert(0, _GRASP_DIR)
sys.path.insert(0, _GRASP_PARENT_DIR)

# cv_bridge는 NumPy 1.x 기준 컴파일 → NumPy 2.x에서 _ARRAY_API 오류 발생
# detect.py는 cv_bridge를 직접 사용하지 않으므로 mock으로 우회
from unittest.mock import MagicMock
for _cv_mod in ['cv_bridge', 'cv_bridge.boost', 'cv_bridge.boost.cv_bridge_boost']:
    sys.modules.setdefault(_cv_mod, MagicMock())

import config as _gd_config

# grasp_detection.config로 등록해 패키지 내 상대 임포트가 동일 객체를 참조하도록 함
sys.modules['grasp_detection.config'] = _gd_config

# get_package_share_directory mock (grasp_detection 패키지 미설치 대응)
from unittest.mock import patch
with patch('ament_index_python.packages.get_package_share_directory', return_value=_MODEL_DIR):
    from grasp_detection import grasp_detection_node as _gdn

# ── 함수 및 설정 참조 ─────────────────────────────────────────────────────────
_GDN_CLASS     = _gdn.GraspDetectionNode
make_roi       = _GDN_CLASS.make_roi
crop_roi       = _GDN_CLASS.crop_roi
get_class_name = _GDN_CLASS.get_class_name

ROI      = _gd_config.ROI
YOLO_SEG = _gd_config.YOLO_SEG

# config에 없는 속성 기본값
_IMGSZ      = getattr(YOLO_SEG, 'imgsz',      640)
_AGNOSTIC   = getattr(YOLO_SEG, 'agnostic',   YOLO_SEG.agnostic_nms)
_MULTI_LABEL = getattr(YOLO_SEG, 'multi_label', False)

# SELECT 설정 (config에 없으므로 로컬 정의)
from dataclasses import dataclass as _dc

@_dc
class _SelectConfig:
    strategy: str = "conf"   # "conf" | "area" | "center"

SELECT = _SelectConfig()


def detect_yolo_boxes_from_path(model, image_path, imgsz=640):
    """이미지 경로로 YOLO 추론 → [{"bbox":[x1,y1,x2,y2], "conf":float, "cls":int}, ...]"""
    img = cv2.imread(image_path)
    if img is None:
        return []
    _predict_kwargs = dict(
        source=img, imgsz=imgsz,
        conf=YOLO_SEG.conf, iou=YOLO_SEG.iou,
        agnostic_nms=YOLO_SEG.agnostic_nms,
        max_det=YOLO_SEG.max_det,
        save=False, verbose=False,
    )
    try:
        results = model.predict(**_predict_kwargs)
    except RuntimeError:
        # GPU 아키텍처 미지원(sm_120 등) 시 CPU 폴백
        results = model.predict(**_predict_kwargs, device='cpu')
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    candidates = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
        candidates.append({
            "bbox": [x1, y1, x2, y2],
            "conf": float(boxes.conf[i].item()),
            "cls":  int(boxes.cls[i].item()),
        })
    return candidates


def pick_best_detection(candidates, select_cfg, img_shape):
    """후보 중 select_cfg.strategy 기준으로 최적 1개 반환"""
    if not candidates:
        return None
    strategy = getattr(select_cfg, 'strategy', 'conf')
    if strategy == 'area':
        def _score(c):
            x1, y1, x2, y2 = c['bbox']
            return (x2 - x1) * (y2 - y1)
    elif strategy == 'center':
        h, w = img_shape[:2]
        def _score(c):
            x1, y1, x2, y2 = c['bbox']
            return -(((x1+x2)/2 - w/2)**2 + ((y1+y2)/2 - h/2)**2)
    else:  # conf
        _score = lambda c: c['conf']
    return max(candidates, key=_score)


def bbox_geometry(bbox):
    """bbox → centroid, grasp_line, grasp_angle, grasp_width"""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = x2 - x1, y2 - y1
    if w >= h:
        return {"centroid": [cx, cy],
                "grasp_line_xyxy": [x1, cy, x2, cy],
                "grasp_angle_deg": 0.0,
                "grasp_width_px": float(h)}
    else:
        return {"centroid": [cx, cy],
                "grasp_line_xyxy": [cx, y1, cx, y2],
                "grasp_angle_deg": 90.0,
                "grasp_width_px": float(w)}

class RecycleDetect(Node):
    def __init__(self):
        super().__init__("recycle_detect")

        # 모델 로드
        model_path = os.path.join(_MODEL_DIR, YOLO_SEG.model_path)
        self.get_logger().info(f"모델 로딩 중: {model_path}")
        self.model = YOLO(model_path)
        self.model.conf = YOLO_SEG.conf
        self.model.iou = YOLO_SEG.iou
        self.model.agnostic = _AGNOSTIC
        self.model.multi_label = _MULTI_LABEL
        self.model.max_det = YOLO_SEG.max_det
        self.get_logger().info("모델 로딩 완료")

        # 임시 ROI 저장 디렉터리
        import tempfile
        self._tmp_dir = tempfile.mkdtemp(prefix="recycle_detect_")
        self._frame_idx = 0

        # 웹캠
        self.cap = cv2.VideoCapture(0)
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

        # 화면 출력 (GUI 환경이 없으면 무시)
        try:
            cv2.imshow("RecycleDetect", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.get_logger().info("종료")
                self.cap.release()
                cv2.destroyAllWindows()
                rclpy.shutdown()
        except cv2.error:
            pass

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
            imgsz=_IMGSZ,
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
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
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
