"""
recycle_main.py
────────────────────────────────────────────────────────────────────
RealSense → YOLO 실시간 루프 → 탐지되면 자동
  → [1] Gemini: 분류 + 집을 객체 선택
  → [2] Claude P-제어 정렬 루프 (첨부 ai_agent 방식)
  → [3] 로봇 pick & place

실행:
    python recycle_main.py                     # 탐지되면 자동
    python recycle_main.py --mock              # mock 테스트
    python recycle_main.py --auto-interval 5.0 # 5초 쿨다운
────────────────────────────────────────────────────────────────────
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import mimetypes

import anthropic
import cv2
import numpy as np
from dotenv import load_dotenv
import pyrealsense2 as rs

import rclpy
import DR_init

load_dotenv()

# ── API 설정 ──────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TEMP    = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL   = "claude-sonnet-4-5"
SAVE_DIR       = "img"

# ── P-제어 설정 (첨부 코드 그대로) ───────────────────────────────
MAX_ALIGN  = 5
ERR_THRESH = 20    # 허용 오차 (px)
KP         = 0.6   # P-제어 게인

# ── Gemini 시스템 프롬프트 ────────────────────────────────────────
GEMINI_SYSTEM = """
너는 분리배출 도우미이자 로봇 집기 대상 선택 AI다.
입력: 이미지 + YOLO bbox 후보 목록(all_bboxes)
반드시 아래 JSON 형식으로만 답해라. 코드블록·마크다운 없이 순수 JSON만 출력해라.
{
  "image_path": "입력 이미지 경로 그대로",
  "classification_possible": true,
  "object_name": "plastic",
  "selected_bbox_index": 0,
  "selected_bbox_xyxy": [257, 170, 310, 223],
  "selected_center_xy": [283.5, 196.5],
  "needs_human_action": false,
  "human_action_reason": null,
  "robot_action": "SORT",
  "bin_type": "plastic",
  "answer": "플라스틱병입니다. 바로 분리배출 가능합니다."
}
object_name/bin_type: can, paper, plastic, General waste, vinyl
"""

# ── 두산 세팅 ──────────────────────────────────────────────────────
DR_init.__dsr__id    = 'dsr01'
DR_init.__dsr__model = 'e0509'
rclpy.init()
node = rclpy.create_node('recycle_main', namespace='dsr01')
DR_init.__dsr__node = node

from DSR_ROBOT2 import movel, movej, posx, posj, \
    set_velx, set_accx, set_velj, set_accj, \
    ROBOT_MODE_AUTONOMOUS, set_robot_mode, DR_BASE, \
    get_current_posx

set_robot_mode(ROBOT_MODE_AUTONOMOUS)
set_velj(10); set_accj(20)
set_velx(10, 5); set_accx(20, 10)

# ── 캘리브레이션 로드 ──────────────────────────────────────────────
with open("calibration.json") as f:
    _calib = json.load(f)

M       = np.float32(_calib["transform_matrix"])
TABLE_Z = float(_calib["table_z"])
print(f"✅ 캘리브레이션 완료 | TABLE_Z: {TABLE_Z:.1f}mm", flush=True)

# ── 로봇 상수 ─────────────────────────────────────────────────────
LIFT_Z     = TABLE_Z + 150
APPROACH_Z = TABLE_Z + 50

BIN_COORDS = {
    "bottle":        [459.982, -250.000, 404.905, 164.881, 179.983, 164.891],
    "can":           [459.982,  250.000, 404.905, 164.881, 179.983, 164.891],
    "paper":         [269.979, -250.036, 404.880, 169.404, 179.981, 169.421],
    "general":       [269.979,  250.036, 404.880, 169.404, 179.981, 169.421],
    "plastic":       [459.982, -250.000, 404.905, 164.881, 179.983, 164.891],
    "vinyl":         [269.979,  250.036, 404.880, 169.404, 179.981, 169.421],
    "General waste": [269.979,  250.036, 404.880, 169.404, 179.981, 169.421],
}

# ── YOLO / ROI 설정 ───────────────────────────────────────────────
from dataclasses import dataclass

@dataclass
class ROIConfig:
    x1_ratio: float = 0.28
    y1_ratio: float = 0.19
    x2_ratio: float = 0.62
    y2_ratio: float = 0.55

ROI = ROIConfig()

class YOLOCfg:
    conf         = 0.05
    iou          = 0.45
    agnostic_nms = True
    max_det      = 10
    save         = False
    verbose      = False

COCO_TO_RECYCLE = {
    39: "plastic", 40: "plastic", 41: "plastic", 75: "plastic",
    73: "paper",   45: "general", 74: "general", 76: "general",
}

def coco_cls_to_recycle(cls_id: int, class_name: str) -> str:
    return COCO_TO_RECYCLE.get(cls_id, class_name)


# ════════════════════════════════════════════════════════════════════
# 좌표 변환 (첨부 코드 그대로)
# ════════════════════════════════════════════════════════════════════
def pixel_to_robot(px: float, py: float):
    pt  = np.float32([[[float(px), float(py)]]])
    res = cv2.perspectiveTransform(pt, M)
    return float(res[0][0][0]), float(res[0][0][1]), TABLE_Z

def robot_to_pixel(robot_x: float, robot_y: float):
    M_inv = cv2.invert(M)[1]
    pt    = np.float32([[[robot_x, robot_y]]])
    res   = cv2.perspectiveTransform(pt, M_inv)
    return int(res[0][0][0]), int(res[0][0][1])

def get_tcp():
    try:
        pos = get_current_posx(DR_BASE)
        return float(pos[0][0]), float(pos[0][1]), float(pos[0][2])
    except Exception:
        return 0.0, 0.0, 0.0

def class_to_bin(class_name: str) -> list:
    name = class_name.lower()
    if any(k in name for k in ["plastic","bottle","pet"]): return BIN_COORDS["plastic"]
    elif any(k in name for k in ["paper","cardboard"]):    return BIN_COORDS["paper"]
    elif any(k in name for k in ["can","metal"]):          return BIN_COORDS["can"]
    else:                                                   return BIN_COORDS["general"]


# ════════════════════════════════════════════════════════════════════
# YOLO 헬퍼
# ════════════════════════════════════════════════════════════════════
def make_roi(image, x1r, y1r, x2r, y2r):
    h, w = image.shape[:2]
    x1, y1 = int(w*x1r), int(h*y1r)
    x2, y2 = int(w*x2r), int(h*y2r)
    x1=max(0,min(x1,w-2)); y1=max(0,min(y1,h-2))
    x2=max(x1+1,min(x2,w-1)); y2=max(y1+1,min(y2,h-1))
    return x1, y1, x2, y2

def crop_roi(img, roi):
    x1, y1, x2, y2 = roi
    return img[y1:y2, x1:x2].copy()

def get_class_name(model, cls_id):
    names = model.names
    if isinstance(names, dict): return names.get(cls_id, str(cls_id))
    if isinstance(names, list) and 0 <= cls_id < len(names): return names[cls_id]
    return str(cls_id)

def run_yolo(model, image):
    return model.predict(
        source=image, conf=YOLOCfg.conf, iou=YOLOCfg.iou,
        agnostic_nms=YOLOCfg.agnostic_nms, max_det=YOLOCfg.max_det,
        save=False, verbose=False)[0]

def extract_detections(model, result, roi) -> List[Dict[str, Any]]:
    x1r, y1r, x2r, y2r = roi
    boxes, masks = result.boxes, result.masks
    if boxes is None or len(boxes) == 0:
        return []
    mask_polygons = masks.xy if (masks is not None and masks.xy is not None) else None
    roi_cx = (x2r - x1r) / 2.0
    roi_cy = (y2r - y1r) / 2.0
    detections = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
        conf   = float(boxes.conf[i].item())
        cls_id = int(boxes.cls[i].item())
        cx_roi = (int(x1) + int(x2)) / 2.0
        cy_roi = (int(y1) + int(y2)) / 2.0
        if mask_polygons is not None and i < len(mask_polygons):
            poly = mask_polygons[i]
            if poly is not None and len(poly) > 0:
                cx_roi = float(np.mean(poly[:, 0]))
                cy_roi = float(np.mean(poly[:, 1]))
        detections.append({
            "idx":        int(i),
            "bbox_xyxy":  [int(x1+x1r), int(y1+y1r), int(x2+x1r), int(y2+y1r)],
            "conf":       float(conf),
            "cls_id":     int(cls_id),
            "class_name": coco_cls_to_recycle(cls_id, get_class_name(model, cls_id)),
            "area":       int(max(1, x2-x1) * max(1, y2-y1)),
            "center":     [float(cx_roi + x1r), float(cy_roi + y1r)],
            "dist2":      float((cx_roi - roi_cx)**2 + (cy_roi - roi_cy)**2),
        })
    return detections

def draw_debug(img, detections, roi, result=None):
    vis = img.copy()
    rx1, ry1, rx2, ry2 = roi
    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
    cv2.putText(vis, "ROI", (rx1, max(ry1-10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.circle(vis, (int((rx1+rx2)/2), int((ry1+ry2)/2)), 6, (255, 0, 0), -1)
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        dcx, dcy = det["center"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(vis, (int(dcx), int(dcy)), 5, (0, 0, 255), -1)
        cv2.putText(vis, f"{det['class_name']} {det['conf']:.2f}",
                    (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    if result:
        action = result.get("robot_action", "")
        color  = (0, 200, 0) if action == "SORT" else (0, 0, 255)
        cv2.putText(vis, f"[{action}] {result.get('bin_type','')}  {result.get('answer','')[:30]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


# ════════════════════════════════════════════════════════════════════
# [1] Gemini — 분류 + 집을 객체 선택
# ════════════════════════════════════════════════════════════════════
def _guess_mime(path):
    mime, _ = mimetypes.guess_type(path)
    if mime: return mime
    return {".jpg":"image/jpeg",".jpeg":"image/jpeg",
            ".png":"image/png"}.get(Path(path).suffix.lower(), "image/jpeg")

def _normalize(data, image_path):
    data = dict(data)
    data.setdefault("image_path",              image_path)
    data.setdefault("classification_possible", False)
    data.setdefault("object_name",             None)
    data.setdefault("selected_center_xy",      None)
    data.setdefault("needs_human_action",      True)
    data.setdefault("robot_action",            "STOP")
    data.setdefault("bin_type",                None)
    data.setdefault("answer",                  "판단 어렵습니다.")
    return data

def call_gemini(image_path: str, detections: list, mock: bool = False) -> dict:
    if mock:
        det = min(detections, key=lambda d: d.get("dist2", 1e18)) if detections else {}
        return _normalize({
            "classification_possible": True,
            "object_name":             "plastic",
            "selected_bbox_index":     det.get("idx"),
            "selected_bbox_xyxy":      det.get("bbox_xyxy"),
            "selected_center_xy":      det.get("center"),
            "needs_human_action":      False,
            "robot_action":            "SORT",
            "bin_type":                "plastic",
            "answer":                  "[MOCK] 플라스틱으로 판단됩니다.",
        }, image_path)

    from google import genai
    from google.genai import types
    client    = genai.Client(api_key=GEMINI_API_KEY)
    img_bytes = Path(image_path).read_bytes()
    contents  = [
        types.Part.from_bytes(data=img_bytes, mime_type=_guess_mime(image_path)),
        json.dumps({"image_path": image_path, "all_bboxes": detections},
                   ensure_ascii=False, indent=2),
    ]
    response = client.models.generate_content(
        model=GEMINI_MODEL, contents=contents,
        config=types.GenerateContentConfig(
            temperature=GEMINI_TEMP,
            system_instruction=GEMINI_SYSTEM,
            response_mime_type="application/json",
        ),
    )
    text = (response.text or "").strip()
    if not text: raise RuntimeError("Gemini 응답 비어있음")
    return _normalize(json.loads(text), image_path)


# ════════════════════════════════════════════════════════════════════
# [2] Claude — P-제어 정렬 (첨부 ai_agent 방식 그대로)
# ════════════════════════════════════════════════════════════════════
claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

def ask_claude_position(frame: np.ndarray, roi: tuple) -> dict:
    """
    현재 프레임 ROI → Claude → 물체 픽셀 좌표 반환
    첨부 코드 ask_claude() 방식 그대로
    """
    x1, y1, x2, y2 = roi
    roi_img = frame[y1:y2, x1:x2]
    h, w    = roi_img.shape[:2]

    cv2.imwrite(os.path.join(SAVE_DIR, "last_roi.jpg"), roi_img)

    _, buf = cv2.imencode(".jpg", roi_img)
    b64    = base64.b64encode(buf).decode("utf-8")

    resp = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": b64,
                }},
                {"type": "text", "text": f"""Top-view robot table image. Size: {w}x{h}px.

Find ONE recyclable waste object on the WHITE TABLE SURFACE ONLY.
Valid: cans, bottles, cups, boxes lying on table.
Invalid: table surface, holes, markings, objects outside table.

JSON only:
{{
  "found": true/false,
  "object": "bottle/can/paper/general",
  "pixel_x": center x (0~{w}),
  "pixel_y": center y (0~{h}),
  "reason": "brief description"
}}"""},
            ],
        }],
    )

    text = resp.content[0].text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"): text = text[4:]
    result = json.loads(text.strip())

    # ROI 크롭 기준 → 전체 이미지 픽셀 변환
    if result.get("found"):
        result["pixel_x"] += x1
        result["pixel_y"] += y1
        print(f"  Claude: {result['object']} @ pixel({result['pixel_x']},{result['pixel_y']}) — {result['reason']}", flush=True)
    else:
        print(f"  Claude: 물체 없음", flush=True)
    return result


def capture_frame(pipeline_rs) -> np.ndarray:
    """RealSense에서 최신 프레임 1장 캡처"""
    for _ in range(3):   # 최신 프레임 3장 버리기
        pipeline_rs.wait_for_frames(timeout_ms=3000)
    frames = pipeline_rs.wait_for_frames(timeout_ms=5000)
    color  = frames.get_color_frame()
    if not color:
        return None
    return np.asanyarray(color.get_data())


def save_debug_align(frame, px, py, gripper_px, gripper_py, err_x, err_y, tag):
    """정렬 디버그 이미지 저장"""
    img = frame.copy()
    cv2.circle(img, (px, py), 18, (0, 0, 255), 3)
    cv2.drawMarker(img, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 35, 3)
    cv2.circle(img, (gripper_px, gripper_py), 12, (0, 220, 0), 3)
    cv2.drawMarker(img, (gripper_px, gripper_py), (0, 220, 0), cv2.MARKER_CROSS, 35, 3)
    cv2.line(img, (gripper_px, gripper_py), (px, py), (0, 200, 0), 2)
    cv2.putText(img, f"err:({err_x:+d},{err_y:+d})px",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
    path = os.path.join(SAVE_DIR, f"{tag}.jpg")
    cv2.imwrite(path, img)
    print(f"  📸 {path}", flush=True)


# ════════════════════════════════════════════════════════════════════
# [3] 로봇 pick & place with Claude P-제어 정렬
# ════════════════════════════════════════════════════════════════════
# ── 고정 집기 좌표 (절대 좌표) ──────────────────────────────────
FIXED_GRASP_POS = [570.619, -0.001, 149.84, 0.002, -179.219, 0.003]

def run_pick_and_place(gemini_result: dict, pipeline_rs, roi: tuple, mock: bool):
    bin_type = gemini_result.get("bin_type") or "general"
    bin_pos  = class_to_bin(bin_type)

    # 고정 좌표에서 P-제어 보정용 초기값 추출
    cur_x = FIXED_GRASP_POS[0]
    cur_y = FIXED_GRASP_POS[1]
    cur_z = FIXED_GRASP_POS[2]

    print(f"\n  고정 집기 좌표: ({cur_x},{cur_y},{cur_z})", flush=True)
    print(f"  분류: {bin_type} → bin({bin_pos[0]:.0f},{bin_pos[1]:.0f})", flush=True)

    # 1. 홈
    print("1. 홈", flush=True)
    movej(posj(0, 0, 90, 0, 90, 0))
    time.sleep(2)

    # 2. 고정 좌표로 바로 이동
    print(f"2. 고정 좌표 이동 → {FIXED_GRASP_POS}", flush=True)
    movel(posx(*FIXED_GRASP_POS), ref=DR_BASE)
    time.sleep(1.5)

    ax, ay, az = get_tcp()
    print(f"   실제 TCP: ({ax:.1f},{ay:.1f},{az:.1f})mm", flush=True)

    # ── Claude P-제어 정렬 루프 (첨부 코드 방식 그대로) ──────────
    print(f"\n📷 Claude P-제어 정렬 시작 (최대 {MAX_ALIGN}회)", flush=True)
    aligned = False
    obj_px, obj_py = int(px_init), int(py_init)

    for align in range(1, MAX_ALIGN + 1):
        print(f"\n  [{align}/{MAX_ALIGN}] 카메라 캡처 + Claude 판단...", flush=True)

        if mock:
            # mock: 현재 그리퍼 위치 = 목표 (바로 정렬 완료)
            gripper_px, gripper_py = robot_to_pixel(cur_x, cur_y)
            check = {"found": True, "object": bin_type,
                     "pixel_x": obj_px, "pixel_y": obj_py}
        else:
            check_frame = capture_frame(pipeline_rs)
            if check_frame is None:
                print(f"  [{align}] 프레임 없음", flush=True)
                continue
            check = ask_claude_position(check_frame, roi)

        if not check.get("found"):
            print(f"  [{align}] Claude: 물체 못 찾음", flush=True)
            continue

        obj_px     = check["pixel_x"]
        obj_py     = check["pixel_y"]
        gripper_px, gripper_py = robot_to_pixel(cur_x, cur_y)

        err_x = obj_px - gripper_px
        err_y = obj_py - gripper_py

        rx, ry, rz = get_tcp()
        print(f"  물체픽셀:({obj_px},{obj_py})  그리퍼픽셀:({gripper_px},{gripper_py})  오차:({err_x:+d},{err_y:+d})px", flush=True)
        print(f"  실제TCP:({rx:.1f},{ry:.1f})", flush=True)

        if not mock:
            save_debug_align(check_frame, obj_px, obj_py,
                             gripper_px, gripper_py, err_x, err_y,
                             f"align_{align}")

        if abs(err_x) <= ERR_THRESH and abs(err_y) <= ERR_THRESH:
            print(f"  ✅ 정렬 완료!", flush=True)
            aligned = True
            break

        # P-제어 보정
        new_x, new_y, _ = pixel_to_robot(obj_px, obj_py)
        cur_x += (new_x - cur_x) * KP
        cur_y += (new_y - cur_y) * KP
        print(f"  🔧 P보정 → 로봇({cur_x:.1f},{cur_y:.1f})", flush=True)

        movel(posx(cur_x, cur_y, cur_z, 0, 180, 0), ref=DR_BASE)
        time.sleep(1.2)

    if not aligned:
        print(f"  ⚠️ 정렬 미완 → 현재 위치로 집기", flush=True)

    # 3. 하강 → 집기
    print(f"\n3. 하강 → z={cur_z:.1f}mm", flush=True)
    movel(posx(cur_x, cur_y, cur_z, 0, 180, 0), ref=DR_BASE)
    time.sleep(1.2)

    gx, gy, gz = get_tcp()
    print(f"   집기 직전 TCP: ({gx:.1f},{gy:.1f},{gz:.1f})mm", flush=True)

    print("4. 집기 (그리퍼 닫기)", flush=True)
    # gripper_close() ← 그리퍼 서비스 연결 시 활성화
    time.sleep(0.6)

    # 4. Lift
    print(f"5. Lift → z={LIFT_Z:.1f}mm", flush=True)
    movel(posx(cur_x, cur_y, LIFT_Z, 0, 180, 0), ref=DR_BASE)
    time.sleep(1.5)

    # 5. 분리수거함
    print(f"6. 분리수거함 → ({bin_pos[0]:.0f},{bin_pos[1]:.0f})", flush=True)
    movel(posx(*bin_pos), ref=DR_BASE)
    time.sleep(2.0)

    print("7. 놓기 (그리퍼 열기)", flush=True)
    # gripper_open() ← 그리퍼 서비스 연결 시 활성화
    time.sleep(0.5)

    # 6. 홈 복귀
    print("8. 홈 복귀", flush=True)
    movej(posj(0, 0, 0, 0, 0, 0))
    time.sleep(2)

    print(f"\n✅ 완료! ({bin_type})  최종 로봇좌표: ({cur_x:.1f},{cur_y:.1f})", flush=True)


# ════════════════════════════════════════════════════════════════════
# 메인 루프
# ════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         type=str,   default="yolov8n-seg.pt")
    parser.add_argument("--trigger-key",   type=str,   default="g")
    parser.add_argument("--auto-interval", type=float, default=3.0,
                        help="탐지 후 자동 호출 쿨다운(초) 기본 3초")
    parser.add_argument("--no-preview",    action="store_true")
    parser.add_argument("--mock",          action="store_true")
    parser.add_argument("--width",         type=int,   default=640)
    parser.add_argument("--height",        type=int,   default=480)
    parser.add_argument("--fps",           type=int,   default=30)
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    # YOLO 로드
    print(f"[INFO] YOLO 모델 로딩: {args.model}", file=sys.stderr)
    from ultralytics import YOLO
    yolo = YOLO(args.model)

    # RealSense
    pipeline_rs = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    pipeline_rs.start(cfg)
    for _ in range(30):
        pipeline_rs.wait_for_frames()
    print(f"[RealSense] {args.width}x{args.height}@{args.fps}fps 시작", file=sys.stderr)

    trigger_key    = ord(args.trigger_key[0])
    last_call_time = 0.0
    latest_frame   = None
    latest_dets    = []
    latest_roi     = None
    last_result    = None
    robot_busy     = False

    print(f"[INFO] 실행 중 | '{args.trigger_key}':수동 / 탐지시 자동 / 'q':종료", file=sys.stderr)

    try:
        while True:
            # 프레임 읽기
            frames = pipeline_rs.wait_for_frames(timeout_ms=5000)
            color  = frames.get_color_frame()
            if not color:
                continue
            frame = np.asanyarray(color.get_data())

            # YOLO
            roi     = make_roi(frame, ROI.x1_ratio, ROI.y1_ratio, ROI.x2_ratio, ROI.y2_ratio)
            roi_img = crop_roi(frame, roi)
            result  = run_yolo(yolo, roi_img)
            dets    = extract_detections(yolo, result, roi)

            latest_frame = frame.copy()
            latest_dets  = dets
            latest_roi   = roi

            now         = time.time()
            should_call = False

            # 탐지되면 자동 (쿨다운 적용)
            if dets and not robot_busy:
                if (now - last_call_time) >= args.auto_interval:
                    should_call = True

            # 프리뷰
            if not args.no_preview:
                debug = draw_debug(frame, dets, roi, last_result)
                status = "🤖 BUSY" if robot_busy else f"'{args.trigger_key}':수동 | dets:{len(dets)}"
                cv2.putText(debug, status, (10, debug.shape[0]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                cv2.imshow("YOLO + Gemini + Claude + Robot", debug)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == trigger_key and not robot_busy:
                    should_call = True

            if not should_call or not latest_dets or robot_busy:
                continue

            last_call_time = now

            # 프레임 + 디버그 저장
            ts       = int(now * 1000)
            img_path = os.path.join(SAVE_DIR, f"frame_{ts}.jpg")
            cv2.imwrite(img_path, latest_frame)
            cv2.imwrite(os.path.join(SAVE_DIR, f"yolo_det_{ts}.jpg"),
                        draw_debug(latest_frame, latest_dets, latest_roi))

            # YOLO 결과 출력
            print(f"\n{'='*60}", flush=True)
            print(f"[YOLO] {len(latest_dets)}개  {time.strftime('%H:%M:%S')}", flush=True)
            print(json.dumps({
                "ok": True, "roi_xyxy": list(latest_roi),
                "detections": latest_dets,
            }, ensure_ascii=False, indent=2), flush=True)
            print(f"{'='*60}", flush=True)

            # [1] Gemini 분류
            print(f"\n🔵 [1/2] Gemini 분류 중...", flush=True)
            try:
                t0          = time.time()
                gemini_res  = call_gemini(img_path, latest_dets, mock=args.mock)
                last_result = gemini_res
                print(f"[GEMINI 응답] ({time.time()-t0:.2f}초)", flush=True)
                print(json.dumps(gemini_res, ensure_ascii=False, indent=2), flush=True)
            except Exception as e:
                print(f"❌ Gemini 실패: {e}", flush=True)
                continue

            if gemini_res.get("robot_action") != "SORT":
                print(f"⚠️  STOP → {gemini_res.get('answer','')}", flush=True)
                continue

            # [2] Claude + 로봇
            robot_busy = True
            print(f"\n🧠 [2/2] Claude P-제어 정렬 + 로봇 동작", flush=True)
            print(f"   bin_type: {gemini_res.get('bin_type')}  answer: {gemini_res.get('answer','')}", flush=True)
            try:
                run_pick_and_place(gemini_res, pipeline_rs, tuple(latest_roi), args.mock)
            except Exception as e:
                print(f"❌ 로봇 오류: {e}", flush=True)
            finally:
                robot_busy = False

    except KeyboardInterrupt:
        print("\n[INFO] 종료", file=sys.stderr)
    finally:
        pipeline_rs.stop()
        if not args.no_preview:
            cv2.destroyAllWindows()
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
        print("종료!", flush=True)


if __name__ == "__main__":
    main()