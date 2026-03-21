from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
import numpy as np

def draw_yolo_results(image_path, results, save_path="result_vis.jpg"):
    img = cv2.imread(image_path)
    r = results[0]

    boxes = r.boxes

    if boxes is None:
        print("no boxes")
        return

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
        conf = float(boxes.conf[i])
        cls = int(boxes.cls[i])

        label = f"{model.names[cls]} {conf:.2f}"

        # bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 텍스트
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # 중심점
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

    cv2.imwrite(save_path, img)
    print("저장 완료:", save_path)
    


print("모델 로딩 시작")
model = YOLO("models/waste3.pt")
print("모델 로딩 완료")

print("class names:", model.names)
print("추론 시작")
image_name = "./local_Test_grasp_detection/results/debug_roi_test10.jpg"

results = model.predict(image_name, imgsz=640, conf=0.01)

draw_yolo_results(image_name, results, "result_all_boxes.jpg")