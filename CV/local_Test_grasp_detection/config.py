from dataclasses import dataclass


@dataclass
class ROIConfig:
    # 작업대 기준 ROI
    x1_ratio: float = 0.285
    y1_ratio: float = 0.455
    x2_ratio: float = 0.6
    y2_ratio: float = 1.1


@dataclass
class YOLOSegConfig:
    # waste1 모델 경로
    model_path: str = "../models/waste4.pt"
    conf: float = 0.5
    # imgsz: int = 640
    iou: int = 0.45
    agnostic : int = False
    multi_label : int = True
    max_det : int =  1000


@dataclass
class SelectConfig:
    # 중앙보다 "아래쪽 중앙" 물체를 더 강하게 선호
    center_weight: float = 3.2
    area_weight: float = 0.6
    edge_penalty: float = 1.0
    bottom_bonus: float = 0.8
    min_area_px: int = 800
    min_cy_ratio: float = 0.55


ROI = ROIConfig()
YOLO_SEG = YOLOSegConfig()
SELECT = SelectConfig()