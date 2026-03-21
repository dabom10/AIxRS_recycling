from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ROIConfig:
    x1_ratio: float = 0.28
    y1_ratio: float = 0.19
    x2_ratio: float = 0.62
    y2_ratio: float = 0.55


@dataclass
class YOLOSegConfig:
    model_path: str = str(PACKAGE_ROOT / "model" / "yolov8m-seg.pt")
    conf: float = 0.1
    iou: float = 0.45
    agnostic_nms: bool = False
    max_det: int = 1000
    save: bool = False
    verbose: bool = False


@dataclass
class TopicConfig:
    image_topic: str = "/camera/image_raw"
    detection_topic: str = "/grasp_detection/detections"
    debug_image_topic: str = "/grasp_detection/debug_image"
    publish_debug_image: bool = True


ROI = ROIConfig()
YOLO_SEG = YOLOSegConfig()
TOPIC = TopicConfig()
