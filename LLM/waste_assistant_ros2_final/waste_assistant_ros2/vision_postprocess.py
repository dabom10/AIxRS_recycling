import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2


os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')


class SimpleCfg:
    def __init__(self, center_weight=1.5, area_weight=1.0, bottom_bonus=0.6, edge_penalty=0.3, min_cy_ratio=0.2):
        self.center_weight = center_weight
        self.area_weight = area_weight
        self.bottom_bonus = bottom_bonus
        self.edge_penalty = edge_penalty
        self.min_cy_ratio = min_cy_ratio


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


def detect_yolo_boxes_from_path(model, image_path: str, imgsz=416) -> List[Dict[str, Any]]:
    results = model(image_path, size=imgsz)
    predictions = results.pred[0]
    out = []
    if predictions is None or len(predictions) == 0:
        return out
    for pred in predictions:
        x1, y1, x2, y2, conf, cls = pred.tolist()
        out.append({'bbox': [int(x1), int(y1), int(x2), int(y2)], 'conf': float(conf), 'cls': int(cls)})
    return out


def get_class_name(model, cls_id: int) -> str:
    names = model.names
    if isinstance(names, dict):
        return names.get(cls_id, str(cls_id))
    if isinstance(names, list) and 0 <= cls_id < len(names):
        return names[cls_id]
    return str(cls_id)


def draw_all_candidates_on_roi(roi_img, candidates, model, save_path: str) -> None:
    vis = roi_img.copy()
    for item in candidates:
        x1, y1, x2, y2 = item['bbox']
        conf = item['conf']
        cls_id = item['cls']
        label = f"{get_class_name(model, cls_id)} {conf:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(save_path, vis)


def pick_best_detection(candidates, cfg: SimpleCfg, roi_shape) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    h, w = roi_shape[:2]
    cx0 = w / 2.0
    cy0 = h * 0.78
    diag = math.hypot(w, h)
    best = None
    best_score = -1e18
    for item in candidates:
        x1, y1, x2, y2 = item['bbox']
        conf = item['conf']
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area = bw * bh
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if cy < h * cfg.min_cy_ratio:
            continue
        dist = math.hypot(cx - cx0, cy - cy0) / max(diag, 1e-6)
        bottom_bias = cy / float(h)
        score = cfg.center_weight * (1.0 - dist) + cfg.area_weight * (area / float(w * h)) + cfg.bottom_bonus * bottom_bias + 0.8 * conf
        margin = 5
        if x1 <= margin or y1 <= margin or x2 >= (w - margin) or y2 >= (h - margin):
            score -= cfg.edge_penalty
        if score > best_score:
            best_score = score
            best = item
    return best if best is not None else max(candidates, key=lambda x: x['conf'])


def bbox_geometry(bbox):
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    if bh >= bw:
        major_axis_angle_deg = 90.0
        grasp_angle_deg = 0.0
        grasp_line_xyxy = [float(x1), float(cy), float(x2), float(cy)]
        grasp_width_px = float(bw)
    else:
        major_axis_angle_deg = 0.0
        grasp_angle_deg = 90.0
        grasp_line_xyxy = [float(cx), float(y1), float(cx), float(y2)]
        grasp_width_px = float(bh)
    return {
        'centroid': (cx, cy),
        'object_width_px': float(bw),
        'object_height_px': float(bh),
        'major_axis_angle_deg': float(major_axis_angle_deg),
        'grasp_angle_deg': float(grasp_angle_deg),
        'grasp_line_xyxy': grasp_line_xyxy,
        'grasp_width_px': float(grasp_width_px),
    }


def run_postprocess(model, image_path: str, roi_ratios=(0.2, 0.2, 0.8, 0.8), output_dir='results') -> Dict[str, Any]:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    image_name = Path(image_path).stem
    os.makedirs(output_dir, exist_ok=True)

    roi = make_roi(img, *roi_ratios)
    color_roi = crop_roi(img, roi)
    roi_path = str(Path(output_dir) / f'debug_roi_{image_name}.jpg')
    cv2.imwrite(roi_path, color_roi)
    candidates = detect_yolo_boxes_from_path(model, roi_path, imgsz=416)
    all_det_path = str(Path(output_dir) / f'debug_all_{image_name}.jpg')
    draw_all_candidates_on_roi(color_roi, candidates, model, all_det_path)
    selected = pick_best_detection(candidates, SimpleCfg(), color_roi.shape)
    result = {
        'ok': selected is not None,
        'image': image_path,
        'roi_saved_path': roi_path,
        'all_det_image': all_det_path,
        'all_detections': candidates,
    }
    if selected is None:
        return result
    geom = bbox_geometry(selected['bbox'])
    result.update({
        'selected_bbox': selected['bbox'],
        'selected_conf': selected['conf'],
        'selected_class': selected['cls'],
        'geometry': geom,
    })
    return result


if __name__ == '__main__':
    print(json.dumps({'message': 'Import this module from your YOLO pipeline.'}, ensure_ascii=False, indent=2))
