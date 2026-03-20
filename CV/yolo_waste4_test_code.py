import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import cv2
import yolov5

# 1. 모델 로드
model = yolov5.load('models/waste4.pt')
# 2. 파라미터 설정
model.conf = 0.5
model.iou = 0.45
model.agnostic = False
model.multi_label = True
model.max_det = 1000

# 3. 입력 이미지
img_path = "waste_grasp/results/debug_roi.jpg"

# 4. 추론
results = model(img_path, size=416)

# 5. 예측값 가져오기
predictions = results.pred[0]

# 6. 원본 이미지를 writable 하게 읽기
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없음: {img_path}")

draw_img = img.copy()   # writable copy

# 클래스 이름
names = model.names

if predictions is None or len(predictions) == 0:
    print("검출된 객체 없음")
else:
    print("=== detections ===")
    for i, pred in enumerate(predictions):
        x1, y1, x2, y2, conf, cls = pred.tolist()

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls = int(cls)
        label = f"{names[cls]} {conf:.2f}"

        print(f"{i}: class={cls} ({names[cls]}), conf={conf:.4f}, box=({x1}, {y1}, {x2}, {y2})")

        # 박스 그리기
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 라벨 그리기
        cv2.putText(
            draw_img,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

# 7. 저장
os.makedirs("results", exist_ok=True)
save_path = "results/debug_roi_detected.jpg"
cv2.imwrite(save_path, draw_img)

print(f"결과 저장 완료: {save_path}")