from ultralytics import YOLO
import cv2
import numpy as np

print("모델 로딩 시작")
model = YOLO("./models/waste1.pt")
print("모델 로딩 완료")

image_name = "./local_Test_grasp_detection/results/debug_roi_test10.jpg"
img = cv2.imread(image_name)

results = model(img, conf=0.1)
r = results[0]

print("class names:", model.names)

overlay = img.copy()

# 🔥 ROI 기준점 (중앙 slightly below)
h, w = img.shape[:2]
target_cx = w // 2
target_cy = int(h * 0.52)

# 중심 기준점 표시 (파랑)
cv2.circle(overlay, (target_cx, target_cy), 8, (255, 0, 0), -1)
cv2.putText(
    overlay,
    "target",
    (target_cx + 10, target_cy),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255, 0, 0),
    2
)

if r.masks is not None:
    masks = r.masks.data.cpu().numpy()
    boxes = r.boxes

    for i, mask in enumerate(masks):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        name = model.names[cls_id]

        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

        # 🔹 bbox 그리기
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 🔹 mask 시각화
        mask_uint8 = (mask * 255).astype(np.uint8)
        colored = np.zeros_like(img)
        colored[:, :, 1] = mask_uint8  # 초록

        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.4, 0)

        # 🔹 mask 중심 계산
        ys, xs = np.where(mask_uint8 > 0)
        if len(xs) == 0:
            continue

        cx = int(np.mean(xs))
        cy = int(np.mean(ys))

        # 🔴 mask 중심
        cv2.circle(overlay, (cx, cy), 6, (0, 0, 255), -1)

        # 🟡 bbox 중심
        bx = int((x1 + x2) / 2)
        by = int((y1 + y2) / 2)
        cv2.circle(overlay, (bx, by), 4, (0, 255, 255), -1)

        # 🔹 거리 계산 (디버깅용)
        dist = (cx - target_cx) ** 2 + (cy - target_cy) ** 2

        label = f"{name} {conf:.2f} A:{(x2-x1)*(y2-y1)} D:{int(dist)}"

        cv2.putText(
            overlay,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

    cv2.imwrite("result_mask_debug.jpg", overlay)
    print("mask + bbox 결과 저장 완료")

else:
    print("mask 없음")