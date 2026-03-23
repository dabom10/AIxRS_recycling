"""
YOLO ROI 객체 탐지 노드
- ROI 영역 안에서만 YOLO 탐지
- 탐지된 물체에 바운딩박스 그려서 저장
- 실시간 확인용
"""
import cv2
import json
import numpy as np
from ultralytics import YOLO
from camera_node import CameraNode

# ── ROI 로드 ──────────────────────────────────
with open("roi_config.json") as f:
    _roi = json.load(f)["roi"]
ROI = (_roi["x1"], _roi["y1"], _roi["x2"], _roi["y2"])
print(f"✅ ROI: {ROI}")

# ── YOLO 로드 ─────────────────────────────────
yolo = YOLO("yolov8n.pt")
print("✅ YOLO 로드 완료\n")

def detect_and_draw(image: np.ndarray) -> np.ndarray:
    """
    ROI 안에서만 YOLO 탐지 후
    전체 이미지에 바운딩박스 그려서 반환
    """
    x1, y1, x2, y2 = ROI
    img = image.copy()

    # ROI 박스 표시 (노란색)
    cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,255),2)
    cv2.putText(img, "ROI", (x1+5,y1+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # ROI 크롭 후 YOLO 탐지
    roi_img = image[y1:y2, x1:x2]
    results  = yolo(roi_img, conf=0.2, verbose=False)

    if not results[0].boxes:
        cv2.putText(img, "NO DETECTION", (x1+10, y1+50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        return img

    # 탐지된 물체마다 바운딩박스
    for box in results[0].boxes:
        label = results[0].names[int(box.cls)]
        conf  = float(box.conf)
        bx1, by1, bx2, by2 = box.xyxy[0]

        # ROI 기준 → 전체 이미지 기준으로 변환
        gx1 = int(bx1) + x1
        gy1 = int(by1) + y1
        gx2 = int(bx2) + x1
        gy2 = int(by2) + y1

        # 바운딩박스 (초록)
        cv2.rectangle(img, (gx1,gy1),(gx2,gy2),(0,255,0), 2)

        # 물체 중심점 (빨간 점)
        cx = (gx1 + gx2) // 2
        cy = (gy1 + gy2) // 2
        cv2.circle(img, (cx,cy), 5, (0,0,255), -1)

        # 라벨 + confidence
        cv2.putText(img, f"{label} {conf:.2f}",
                    (gx1, gy1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        print(f"  감지: {label} ({conf:.2f}) | 중심: ({cx},{cy})")

    return img

# ── 메인 ──────────────────────────────────────
if __name__ == "__main__":

    print("📷 카메라 초기화...")
    cam = CameraNode()

    print("🔍 YOLO 탐지 시작\n")

    while True:
        color, _, _ = cam.get_frames()

        print("─" * 40)
        result_img = detect_and_draw(color)

        # 결과 저장
        cv2.imwrite("yolo_result.jpg", result_img)
        print("  📸 yolo_result.jpg 저장")

        key = input("\nEnter: 다시 탐지 | q: 종료 > ").strip().lower()
        if key == 'q':
            break

    cam.release()
    print("종료!")