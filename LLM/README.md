# Waste Robot Project

Gemini 멀티모달 입력(이미지 + 음성 + 질문)과 YOLO 후처리 결과를 결합해서,
로봇이 `NO_MOVE_INSTRUCT / NO_MOVE_ASK / NO_MOVE_HUMAN / MOVE_CAN / MOVE_PAPER / MOVE_PLASTIC / MOVE_GENERAL`
중 하나의 안전한 행동만 수행하도록 설계한 ROS2 예제 프로젝트입니다.

## 구성
- `waste_sort_interfaces`: 커스텀 Action/Message 인터페이스
- `waste_assistant_ros2`: Gemini 호출, 결정 게이트, ROS2 노드, YOLO 후처리 유틸

## 핵심 정책
- 라벨 제거, 세척, 내용물 비우기 등 전처리가 필요하면 **무조건 no-move**
- 비전 결과와 LLM 결과가 불일치하면 **질문 또는 안내**
- 충분히 확신이 높을 때만 **지정 분리수거함으로 이동**

## 빠른 시작
1. `waste_assistant_ros2/.env.example`를 복사해 `.env` 생성
2. ROS2 워크스페이스 `src/`에 두 패키지를 복사
3. 의존성 설치
   - `pip install google-genai python-dotenv pydantic opencv-python`
4. 빌드
   - `colcon build --packages-select waste_sort_interfaces waste_assistant_ros2`
   - `source install/setup.bash`
5. 실행
   - `ros2 launch waste_assistant_ros2 waste_system.launch.py`

## 입력 예시
`/waste/query` 토픽에 JSON 문자열 publish

```json
{
  "request_id": "req-0001",
  "image_path": "/abs/path/to/plastic_bottle.jpg",
  "audio_path": "/abs/path/to/question.wav",
  "question": "이거 그대로 버려도 돼?",
  "vision_hint": {
    "object": "plastic_bottle",
    "subtype": "transparent_pet",
    "confidence": 0.93,
    "has_label": true,
    "dirty": false,
    "liquid_inside": false
  }
}
```

## 참고
YOLO 후처리 로직은 업로드된 설명 파일의 구조를 반영해 ROI 저장 후 파일 경로 기준으로 재추론하고,
전체 후보 저장 → best detection 선택 흐름으로 구현했습니다. fileciteturn2file0
