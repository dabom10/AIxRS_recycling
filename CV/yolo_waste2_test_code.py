from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("models/waste2.pt")
image_name= "waste_grasp/debug_roi.jpg"
img = cv2.imread(image_name)

results = model(img, conf=0.01)
r = results[0]

if r.masks is None:
    print("no object")
    exit()

masks = r.masks.data.cpu().numpy()

best_score = -1
best_center = None

H, W = img.shape[:2]
center_x, center_y = W // 2, H // 2

for mask in masks:
    mask_uint8 = (mask * 255).astype(np.uint8)

    ys, xs = np.where(mask_uint8 > 0)
    if len(xs) == 0:
        continue

    cx = int(np.mean(xs))
    cy = int(np.mean(ys))

    area = len(xs)

    # 🔥 중심과 거리 계산
    dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)

    # 🔥 점수 = 중앙 가까움 + 면적 큼
    score = area - dist * 50

    if score > best_score:
        best_score = score
        best_center = (cx, cy)
        best_mask = mask_uint8

# 🔥 시각화
vis = img.copy()

# mask 칠하기
colored = np.zeros_like(vis)
colored[:, :, 1] = best_mask
vis = cv2.addWeighted(vis, 1.0, colored, 0.5, 0)

# 중심점
cv2.circle(vis, best_center, 8, (0, 0, 255), -1)

cv2.imwrite("result_final.jpg", vis)

print("선택된 중심:", best_center)