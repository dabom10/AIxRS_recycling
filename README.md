# AIxRS_recycling
2026 AI X RS project


# Grasp Detection ROS2 Node (file path : CV/grasp_detection)

YOLOv5 기반으로 작업대 ROI 영역에서 객체를 검출하고,  
후처리를 통해 grasp 대상 1개를 선택한 뒤  
중심점, bbox, grasp angle, grasp width 등의 정보를 ROS2 토픽으로 publish하는 노드입니다.

---

## 1. 개요

이 노드는 카메라 이미지에서 관심 영역(ROI)을 잘라낸 뒤 YOLOv5로 객체를 검출합니다.  
검출된 후보들 중 후처리 로직을 통해 집기 적합한 객체 1개를 선택하고,  
해당 객체의 중심점, 바운딩 박스, grasp angle, grasp width 정보를 계산하여 publish합니다.

디버깅을 위해 다음을 함께 지원합니다.

- ROI 내부 전체 후보 박스 시각화
- 최종 선택된 객체와 grasp line 시각화
- 검출 결과 JSON publish
- 디버그 이미지 publish

---

## 2. 프로젝트 구조

```bash
grasp_detection/
├── package.xml
├── setup.py
├── setup.cfg
├── resource/
│   └── grasp_detection
├── launch/
│   └── grasp_detection.launch.py
├── models/
│   └── waste4.pt
├── grasp_detection/
│   ├── __init__.py
│   ├── config.py
│   └── grasp_detection_node.py
└── README.md
````

---

## 3. 토픽 구조

### 입력

```bash
/camera/color/image_raw
```

카메라 원본 이미지 토픽을 subscribe 합니다.

### 출력 JSON

```bash
/grasp_detection/result
```

검출 결과를 JSON 문자열 형태로 publish 합니다.

### 출력 디버그 이미지

```bash
/grasp_detection/debug_image
```

선택된 bbox, centroid, grasp line이 그려진 디버그 이미지를 publish 합니다.

---

## 4. 출력 데이터 예시

`/grasp_detection/result` 토픽에는 아래와 같은 형태의 JSON 문자열이 publish 됩니다.

```json
{
  "ok": true,
  "frame_id": 12,
  "roi_xyxy": [612, 611, 1288, 1296],
  "yolo_bbox_xyxy": [874, 982, 1062, 1288],
  "bbox_xyxy": [874, 982, 1062, 1288],
  "centroid_uv": [968.0, 1135.0],
  "object_width_px": 188.0,
  "object_height_px": 306.0,
  "major_axis_angle_deg": 90.0,
  "grasp_angle_deg": 0.0,
  "grasp_width_px": 188.0,
  "grasp_line_xyxy": [874.0, 1135.0, 1062.0, 1135.0],
  "selected_conf": 0.91,
  "selected_class": 0,
  "selected_class_name": "plastic",
  "all_detections": [
    {
      "bbox": [10, 20, 100, 200],
      "conf": 0.88,
      "cls": 0
    }
  ]
}
```

---

## 5. 주요 처리 흐름

1. `/camera/color/image_raw` 이미지 수신
2. 작업대 기준 ROI 생성
3. ROI 이미지 crop
4. YOLOv5로 ROI 내부 객체 검출
5. 후보 박스 중 최적 객체 1개 선택
6. bbox 기반 centroid, grasp angle, grasp width 계산
7. JSON 결과 publish
8. 디버그 이미지 publish

---

## 6. 모델 파일 위치

모델은 패키지 루트의 `models/` 폴더에 둡니다.

```bash
grasp_detection/models/waste4.pt
```

### 왜 이렇게 두는가

ROS2에서는 `ros2 run`, `ros2 launch` 실행 위치가 달라질 수 있어서
`../models/waste4.pt` 같은 상대경로는 신뢰하기 어렵습니다.

따라서 모델은 패키지에 포함시키고, 런타임에는
`get_package_share_directory("grasp_detection")`를 통해 절대경로로 읽는 방식을 사용합니다.

예시:

```python
from ament_index_python.packages import get_package_share_directory
import os

pkg_path = get_package_share_directory("grasp_detection")
model_path = os.path.join(pkg_path, "models", "waste4.pt")
```

---

## 7. 설정 파일

### `config.py`

ROI, YOLO, 선택 로직 관련 설정값을 관리합니다.

예시 항목:

* ROI 범위 비율
* YOLO 모델 파일명
* confidence threshold
* IOU threshold
* multi_label 여부
* 후보 선택 가중치

예시:

```python
from dataclasses import dataclass


@dataclass
class ROIConfig:
    x1_ratio: float = 0.285
    y1_ratio: float = 0.455
    x2_ratio: float = 0.6
    y2_ratio: float = 1.1


@dataclass
class YOLOSegConfig:
    model_path: str = "waste4.pt"
    conf: float = 0.5
    iou: float = 0.45
    agnostic: bool = False
    multi_label: bool = True
    max_det: int = 1000
    imgsz: int = 416


@dataclass
class SelectConfig:
    center_weight: float = 3.2
    area_weight: float = 0.6
    edge_penalty: float = 1.0
    bottom_bonus: float = 0.8
    min_area_px: int = 800
    min_cy_ratio: float = 0.55


ROI = ROIConfig()
YOLO_SEG = YOLOSegConfig()
SELECT = SelectConfig()
```

---

## 8. 실행 전 설정

### Python 패키지 설치

```bash
pip install yolov5 opencv-python
```

### ROS2 패키지 설치

```bash
sudo apt install ros-humble-cv-bridge
sudo apt install ros-humble-launch ros-humble-launch-ros
```

환경에 따라 `humble` 대신 `foxy`, `iron` 등을 사용해야 할 수 있습니다.

---

## 9. package.xml 의존성

`package.xml`에는 아래 의존성을 포함합니다.

```xml
<depend>rclpy</depend>
<depend>sensor_msgs</depend>
<depend>std_msgs</depend>
<depend>cv_bridge</depend>
<depend>launch</depend>
<depend>launch_ros</depend>
```

---

## 10. setup.py 설정

모델 파일과 launch 파일이 install 되도록 설정해야 합니다.

예시:

```python
from setuptools import setup
from glob import glob

package_name = "grasp_detection"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}", ["README.md"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/models", glob("models/*.pt")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="your_name",
    maintainer_email="you@example.com",
    description="YOLOv5-based grasp detection ROS2 node",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "grasp_detection_node = grasp_detection.grasp_detection_node:main",
        ],
    },
)
```

---

## 11. launch 파일 예시

`launch/grasp_detection.launch.py`

```python
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="grasp_detection",
            executable="grasp_detection_node",
            name="grasp_detection_node",
            output="screen",
            parameters=[
                {
                    "image_topic": "/camera/color/image_raw",
                    "result_topic": "/grasp_detection/result",
                    "debug_topic": "/grasp_detection/debug_image",
                    "debug_roi_dir": "/tmp/grasp_detection_roi",
                    "save_debug_roi": True,
                }
            ]
        )
    ])
```

---

## 12. 빌드 방법

워크스페이스 예시:

```bash
ros2_ws/
└── src/
    └── grasp_detection/
```

빌드:

```bash
cd ~/ros2_ws
colcon build --packages-select grasp_detection
```

환경 적용:

```bash
source install/setup.bash
```

---

## 13. 실행 방법

### run으로 실행

```bash
ros2 run grasp_detection grasp_detection_node
```

### launch로 실행

```bash
ros2 launch grasp_detection grasp_detection.launch.py
```

---

## 14. 결과 확인

### 결과 JSON 확인

```bash
ros2 topic echo /grasp_detection/result
```

### 디버그 이미지 확인

```bash
rqt_image_view
```

이후 `/grasp_detection/debug_image` 토픽을 선택하면 됩니다.

---

## 15. 노드 파라미터

주요 파라미터 예시는 아래와 같습니다.

* `image_topic`: 입력 이미지 토픽
* `result_topic`: 결과 JSON 출력 토픽
* `debug_topic`: 디버그 이미지 출력 토픽
* `debug_roi_dir`: ROI 임시 저장 경로
* `save_debug_roi`: ROI 이미지 저장 여부

필요 시 launch 파일에서 변경할 수 있습니다.

---

## 16. 구현 특징

이 노드는 다음 특징을 가집니다.

* ROI 기반 검출로 불필요한 배경 제거
* YOLO 후보들 중 후처리 로직으로 최적 객체 선택
* bbox 기반 grasp geometry 계산
* ROS2 토픽 기반 결과 전달
* 디버그 이미지 publish 지원
* 패키지 내부 모델 파일 절대경로 로딩 지원

---

## 17. 주의사항

현재 구현은 단독 테스트 환경과 동일한 추론 결과를 최대한 맞추기 위해
ROI 이미지를 파일로 저장한 뒤, 파일 경로 기반으로 YOLO 추론을 수행하는 구조를 유지할 수 있습니다.

즉 현재 처리 흐름은 다음과 같습니다.

1. ROI crop
2. ROI 이미지 저장
3. 저장된 ROI 파일 경로를 YOLO 입력으로 사용

이 방식은 디버깅과 재현성 측면에서는 유리하지만,
실시간 운영 환경에서는 디스크 I/O로 인해 성능 저하가 발생할 수 있습니다.

운영 단계에서는 아래와 같이 분리하는 것을 권장합니다.

* **디버그 모드**: 파일 저장 후 경로 기반 추론
* **운영 모드**: numpy 배열 직접 추론

---

## 18. 향후 개선 방향

* custom message 타입으로 결과 구조화
* numpy 배열 직접 추론 모드 추가
* launch 파라미터 확장
* depth 정보와 결합하여 실제 grasp pose 계산
* 멀티스레드/비동기 구조로 처리량 개선
* GPU 환경 최적화

---

## 19. 참고

* ROS2 Python (`rclpy`)
* OpenCV (`cv2`)
* YOLOv5 Python wrapper
* cv_bridge
* ament_index_python
