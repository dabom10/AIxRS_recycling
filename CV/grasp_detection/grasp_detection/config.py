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
    model_path: str = "waste4.pt"
    conf: float = 0.5
    iou: float = 0.45
    agnostic: bool = False
    multi_label: bool = True
    max_det: int = 1000
    imgsz: int = 416


@dataclass
class SelectConfig:
    # 중앙보다 아래쪽 중앙 물체를 더 선호
    center_weight: float = 3.2
    area_weight: float = 0.6
    edge_penalty: float = 1.0
    bottom_bonus: float = 0.8
    min_area_px: int = 800
    min_cy_ratio: float = 0.55


ROI = ROIConfig()
YOLO_SEG = YOLOSegConfig()
SELECT = SelectConfig()