# [시스템 플로우 시안]

###   - 물체 종류 인식 (유리병, 플라스틱 등) 부분의 폴더
```
시민이 쓰레기를 들고 카메라 앞에 서서 말함
        ↓
"이 더러운 병 버려도 돼요?"
        ↓
[멀티모달 입력]
  - STT: 음성 → 텍스트
  - Vision: 카메라로 객체 촬영
        ↓
[LLM 판단] (AI 파트 담당)
  - 물체 종류 인식 (유리병, 플라스틱 등)
  - 오염도 판단 (더럽다 / 괜찮다)
  - 재질 복합 여부 (라벨 붙어있나 등)
        ↓
      판단 결과
     /         \
  OK            NG (세척 필요 / 분리 필요)
   |                      |
로봇이 받아서         LLM이 말로 안내
분류통에 넣기          + 씻는 모션 시연
                       + 분류통에 넣기
```


:arrows_counterclockwise: 전체 시스템 플로우 (확정 아님)
1단계 — 시민 입력
시민이 쓰레기를 들고 카메라 앞에 서서 말함
"이 페트병 버려도 돼요?"
        ↓
[로보틱스 파트]
- RealSense RGB 캡처
- RealSense Depth 캡처
- 마이크로 음성 수집 → STT → 텍스트 변환

2단계 — AI 파트로 전송 (Request)
[로보틱스 → AI FastAPI 서버]

POST /classify
{
  "rgb_image": "base64...",   // RGB 이미지
  "speech_text": "이 페트병 버려도 돼요?"  // STT 텍스트
}

※ depth는 전송 안 함 — 로보틱스가 로컬에서 처리

3단계 — AI 파트 처리
[AI 파트 내부]

RGB + 텍스트
    ↓
Vision 모델 → 객체 인식 → 픽셀 좌표 (u, v) 추출
    ↓
LLM → 분류 판단
  - 재질: 플라스틱
  - 오염도: 오염됨
  - 판단: 세척 필요
    ↓
TTS → 응답 음성 생성
"세척 후 플라스틱 통에 버려주세요!"

4단계 — 로보틱스 파트로 전송 (Response)
[AI → 로보틱스]

{
  "label": "plastic",
  "category": "washable",
  "action": "wash_and_sort",
  "pixel_x": 320,
  "pixel_y": 240,
  "response_text": "세척 후 플라스틱 통에 버려주세요!"
}

5단계 — 로보틱스 파트 처리
[로보틱스 파트 내부]

pixel_x, pixel_y 수신
    ↓
depth[pixel_y, pixel_x] → Z값 추출
    ↓
카메라 내부 파라미터 (fx, fy, cx, cy) 로 3D 변환
X = (pixel_x - cx) * Z / fx
Y = (pixel_y - cy) * Z / fy
    ↓
Eye-to-hand 캘리브레이션 행렬 T 적용
→ 로봇 베이스 좌표 (X_robot, Y_robot, Z_robot)
    ↓
E0509 역기구학(IK) → 관절 각도 계산
    ↓
모션 실행

6단계 — 모션 실행 (action 기준)
action 값에 따라 분기

"sort"          → 받기 → 해당 분류통 투입
"wash_and_sort" → 받기 → 씻는 모션 → 분류통 투입
"trash"         → 받기 → 일반쓰레기통 투입
"reject"        → 되돌려주기 모션
    ↓
TTS 재생: "세척 후 플라스틱 통에 버려주세요!"

:bar_chart: 한눈에 보는 전체 플로우
시민 (말 + 쓰레기 제시)
        ↓
[로보틱스] RGB 캡처 + STT
        ↓
[AI] 객체인식 + LLM 판단
        ↓ (pixel_x, pixel_y + action)
[로보틱스] depth로 3D 변환
        ↓ (로봇 좌표)
[로보틱스] IK + E0509 모션 실행
        ↓
[로보틱스] TTS 재생