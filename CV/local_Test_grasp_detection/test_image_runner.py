import json
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from config import ROI, YOLO_SEG, PATH


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


def save_roi_preview(img, roi, save_path):
    vis = img.copy()
    x1, y1, x2, y2 = roi
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.imwrite(save_path, vis)


def get_class_name(model, cls_id):
    names = model.names
    if isinstance(names, dict):
        return names.get(cls_id, str(cls_id))
    elif isinstance(names, list) and 0 <= cls_id < len(names):
        return names[cls_id]
    return str(cls_id)


def run_yolov8_seg(model, image):
    results = model.predict(
        source=image,
        conf=YOLO_SEG.conf,
        iou=YOLO_SEG.iou,
        agnostic_nms=YOLO_SEG.agnostic_nms,
        max_det=YOLO_SEG.max_det,
        save=YOLO_SEG.save,
        verbose=YOLO_SEG.verbose,
    )
    return results[0]


def extract_candidates_full_coords(model, result, roi):
    x1r, y1r, _, _ = roi

    boxes = result.boxes
    masks = result.masks

    candidates = []

    if boxes is None or len(boxes) == 0:
        return candidates

    mask_polygons = None
    if masks is not None and masks.xy is not None:
        mask_polygons = masks.xy

    roi_h = roi[3] - roi[1]
    roi_w = roi[2] - roi[0]
    target_cx_roi = roi_w / 2.0
    target_cy_roi = roi_h / 2.0

    print(f"검출 개수: {len(boxes)}")

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
        conf = float(boxes.conf[i].item())
        cls_id = int(boxes.cls[i].item())
        class_name = get_class_name(model, cls_id)

        bw = max(1, int(x2) - int(x1))
        bh = max(1, int(y2) - int(y1))
        area = bw * bh

        # 기본 중심은 bbox 중심 (ROI 기준)
        center_x_roi = (int(x1) + int(x2)) / 2.0
        center_y_roi = (int(y1) + int(y2)) / 2.0

        # mask polygon이 있으면 중심만 polygon 평균으로 보정
        if mask_polygons is not None and i < len(mask_polygons):
            poly = mask_polygons[i]
            if poly is not None and len(poly) > 0:
                center_x_roi = float(np.mean(poly[:, 0]))
                center_y_roi = float(np.mean(poly[:, 1]))

        dist2 = float((center_x_roi - target_cx_roi) ** 2 + (center_y_roi - target_cy_roi) ** 2)

        # 원본 전체 이미지 기준으로 변환
        x1_full = int(x1 + x1r)
        y1_full = int(y1 + y1r)
        x2_full = int(x2 + x1r)
        y2_full = int(y2 + y1r)

        center_x_full = float(center_x_roi + x1r)
        center_y_full = float(center_y_roi + y1r)

        candidates.append({
            "idx": int(i),
            "bbox_xyxy": [x1_full, y1_full, x2_full, y2_full],
            "conf": float(conf),
            "cls_id": int(cls_id),
            "class_name": str(class_name),
            "area": int(area),
            "center": [center_x_full, center_y_full],
            "dist2": float(dist2),
        })

        print(
            f"[{i}] class={class_name}, conf={conf:.4f}, "
            f"bbox_full={[x1_full, y1_full, x2_full, y2_full]}, "
            f"area={area}, center_full=({center_x_full:.1f}, {center_y_full:.1f}), dist2={dist2:.1f}"
        )

    return candidates


def draw_bboxes_on_full_image(img, detections, roi):
    vis = img.copy()

    # ROI도 참고용으로 표시
    rx1, ry1, rx2, ry2 = roi
    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
    cv2.putText(
        vis,
        "ROI",
        (rx1, max(ry1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # ROI 중심도 원본 기준으로 표시
    target_cx = int((rx1 + rx2) / 2)
    target_cy = int((ry1 + ry2) / 2)
    cv2.circle(vis, (target_cx, target_cy), 6, (255, 0, 0), -1)
    cv2.putText(
        vis,
        "target",
        (target_cx + 8, target_cy - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        cx, cy = det["center"]
        class_name = det["class_name"]
        conf = det["conf"]
        area = det["area"]

        # bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 중심점
        cv2.circle(vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)

        label = f"{class_name} {conf:.2f} A:{area}"
        cv2.putText(
            vis,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    return vis


def main():
    image_path = PATH.image_path
    save_dir = PATH.save_dir

    os.makedirs(save_dir, exist_ok=True)

    print("모델 로딩 시작")
    model = YOLO(YOLO_SEG.model_path)
    print("모델 로딩 완료")
    print("class names:", model.names)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없음: {image_path}")

    image_name = Path(image_path).stem

    # 1) ROI 계산
    roi = make_roi(img, ROI.x1_ratio, ROI.y1_ratio, ROI.x2_ratio, ROI.y2_ratio)

    # 2) ROI 미리보기 저장
    roi_preview_path = os.path.join(save_dir, f"roi_preview_{image_name}.png")
    save_roi_preview(img, roi, roi_preview_path)

    # 3) ROI crop
    color_roi = crop_roi(img, roi)

    # 4) crop된 ROI 참고용 저장
    roi_crop_path = os.path.join(save_dir, f"debug_roi_{image_name}.png")
    cv2.imwrite(roi_crop_path, color_roi)

    # 5) ROI만 추론
    print("추론 시작")
    result = run_yolov8_seg(model, color_roi)

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        output = {
            "ok": False,
            "reason": "no_detection",
            "image": str(image_path),
            "roi_xyxy": [
                int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]),
            ],
            "roi_preview_path": str(roi_preview_path),
            "roi_crop_path": str(roi_crop_path),
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    # 6) detection 결과를 원본 전체 좌표계로 변환
    detections = extract_candidates_full_coords(model, result, roi)

    # 7) 원본 이미지 위에 bbox만 시각화
    full_debug_vis = draw_bboxes_on_full_image(img, detections, roi)

    full_debug_path = os.path.join(save_dir, f"debug_full_{image_name}.png")
    cv2.imwrite(full_debug_path, full_debug_vis)
    print("전체 결과 저장:", full_debug_path)

    # 8) 출력도 원본 전체 기준으로만 줌
    output = {
        "ok": True,
        "image": str(image_path),
        "roi_xyxy": [
            int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]),
        ],
        "roi_preview_path": str(roi_preview_path),
        "roi_crop_path": str(roi_crop_path),
        "full_debug_image": str(full_debug_path),
        "target_center_full": [
            float((roi[0] + roi[2]) / 2.0),
            float((roi[1] + roi[3]) / 2.0),
        ],
        "detections": detections
    }

    print("\n=== detections ===")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()