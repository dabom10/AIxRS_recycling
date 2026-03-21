from dataclasses import dataclass


@dataclass
class ROIConfig:
    x1_ratio: float = 0.28
    y1_ratio: float = 0.19
    x2_ratio: float = 0.62
    y2_ratio: float = 0.55


@dataclass
class YOLOSegConfig:
    model_path: str = "./models/yolov8m-seg.pt"
    conf: float = 0.1
    iou: float = 0.45
    agnostic_nms: bool = False
    max_det: int = 1000
    save: bool = False
    verbose: bool = True


@dataclass
class PathConfig:
    image_path: str = "../testData/test7.png"
    save_dir: str = "./local_Test_grasp_detection/results"


ROI = ROIConfig()
YOLO_SEG = YOLOSegConfig()
PATH = PathConfig()