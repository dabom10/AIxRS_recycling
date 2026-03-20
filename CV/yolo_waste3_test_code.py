from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
import numpy as np



print("모델 로딩 시작")
model = YOLO("models/waste3.pt")
print("모델 로딩 완료")

print("class names:", model.names)
print("추론 시작")
image_name= "waste_grasp/debug_roi.jpg"
results = model.predict(image_name, imgsz=640, conf=0.01)

r = results[0]

# segmentation mask 확인
if r.masks is not None:
    masks = r.masks.data.cpu().numpy()

    print("mask 개수:", len(masks))

    for i, mask in enumerate(masks):
        mask = (mask * 255).astype(np.uint8)

        # 중심 좌표 계산
        ys, xs = np.where(mask > 0)

        if len(xs) > 0:
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            print(f"[{i}] center:", cx, cy)

            # 시각화
            img = cv2.imread(image_path)
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), -1)
            cv2.imwrite(f"result_{i}.jpg", img)

else:
    print("mask 없음")