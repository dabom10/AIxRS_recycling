from ultralytics import YOLO
import cv2
import numpy as np

print("모델 로딩 시작")
model = YOLO("./models/waste1.pt")
print("모델 로딩 완료")
image_name= "waste_grasp/debug_roi.jpg"
img = cv2.imread(image_name)

results = model(img, conf=0.1)
r = results[0]

print("class names:", model.names)

if r.masks is not None:
    masks = r.masks.data.cpu().numpy()
    boxes = r.boxes

    overlay = img.copy()

    for i, mask in enumerate(masks):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        name = model.names[cls_id]

        # mask → 컬러
        color = (0, 255, 0)

        mask_uint8 = (mask * 255).astype(np.uint8)
        colored = np.zeros_like(img)
        colored[:, :, 1] = mask_uint8  # 초록색

        # 반투명 overlay
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.5, 0)

        # 중심 좌표 계산
        ys, xs = np.where(mask_uint8 > 0)
        if len(xs) == 0:
            continue

        cx = int(np.mean(xs))
        cy = int(np.mean(ys))

        # 중심점 표시
        cv2.circle(overlay, (cx, cy), 6, (0, 0, 255), -1)

        # 텍스트
        cv2.putText(
            overlay,
            f"{name} {conf:.2f}",
            (cx + 10, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    cv2.imwrite(f"result_mask_{image_name}.jpg", overlay)
    print("mask 결과 저장 완료")

else:
    print("mask 없음")