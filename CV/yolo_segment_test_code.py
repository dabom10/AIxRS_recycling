from ultralytics import YOLO
import cv2
import numpy as np
import os
import json

MODEL_PATH = "./models/yolov8m-seg.pt"
IMAGE_PATH = "./local_Test_grasp_detection/results/debug_roi_test5.jpg"
SAVE_DIR = "./local_Test_grasp_detection/results"

os.makedirs(SAVE_DIR, exist_ok=True)

print("모델 로딩 시작")
model = YOLO(MODEL_PATH)
print("모델 로딩 완료")
print("class names:", model.names)

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"이미지를 읽을 수 없음: {IMAGE_PATH}")

h, w = img.shape[:2]
target_cx = w / 2.0
target_cy = h / 2.0

print("추론 시작")
results = model.predict(
    source=IMAGE_PATH,
    conf=0.01,
    save=False,
    verbose=True
)
r = results[0]

all_vis = img.copy()
mask_vis = img.copy()

candidates = []

boxes = r.boxes
masks = r.masks

if boxes is None or len(boxes) == 0:
    print("검출 결과 없음")
    print(json.dumps({
        "ok": False,
        "reason": "no_detection",
        "image": IMAGE_PATH
    }, ensure_ascii=False, indent=2))
else:
    cv2.circle(all_vis, (int(target_cx), int(target_cy)), 6, (255, 0, 0), -1)
    cv2.putText(
        all_vis,
        "target",
        (int(target_cx) + 8, int(target_cy) - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    print(f"검출 개수: {len(boxes)}")

    mask_polygons = None
    if masks is not None and masks.xy is not None:
        mask_polygons = masks.xy

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
        conf = float(boxes.conf[i].item())
        cls_id = int(boxes.cls[i].item())
        class_name = model.names[cls_id]

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area = bw * bh

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        if mask_polygons is not None and i < len(mask_polygons):
            poly = mask_polygons[i]
            if poly is not None and len(poly) > 0:
                poly_int = np.round(poly).astype(np.int32)

                overlay = mask_vis.copy()
                cv2.fillPoly(overlay, [poly_int], color=(0, 255, 0))
                mask_vis = cv2.addWeighted(mask_vis, 1.0, overlay, 0.35, 0)

                cv2.polylines(mask_vis, [poly_int], isClosed=True, color=(0, 180, 0), thickness=2)

                center_x = float(np.mean(poly[:, 0]))
                center_y = float(np.mean(poly[:, 1]))

        dist2 = (center_x - target_cx) ** 2 + (center_y - target_cy) ** 2

        candidates.append({
            "idx": int(i),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "conf": float(conf),
            "cls_id": int(cls_id),
            "class_name": str(class_name),
            "area": int(area),
            "center": [float(center_x), float(center_y)],
            "dist2": float(dist2),
        })

        cv2.rectangle(all_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        bx = int((x1 + x2) / 2)
        by = int((y1 + y2) / 2)
        cv2.circle(all_vis, (bx, by), 4, (0, 255, 255), -1)

        cv2.circle(all_vis, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)

        label = f"{class_name} {conf:.2f} A:{area}"
        cv2.putText(
            all_vis,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        print(
            f"[{i}] class={class_name}, conf={conf:.4f}, "
            f"bbox={[x1, y1, x2, y2]}, area={area}, "
            f"center=({center_x:.1f}, {center_y:.1f}), dist2={dist2:.1f}"
        )

    final_vis = cv2.addWeighted(all_vis, 0.85, mask_vis, 0.35, 0)

    all_save_path = os.path.join(SAVE_DIR, "hf_yolov8_seg_all_debug.jpg")
    cv2.imwrite(all_save_path, final_vis)
    print("전체 결과 저장:", all_save_path)

    output = {
        "ok": True,
        "image": IMAGE_PATH,
        "all_debug_image": all_save_path,
        "target_center": [target_cx, target_cy],
        "detections": candidates
    }

    print("\n=== detections ===")
    print(json.dumps(output, ensure_ascii=False, indent=2))