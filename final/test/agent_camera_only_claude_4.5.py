import rclpy, DR_init, time, subprocess
import anthropic, base64, json, cv2
import numpy as np
from visualization_msgs.msg import Marker
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from camera_node import CameraNode

# ── 두산 세팅 ─────────────────────────────────
DR_init.__dsr__id    = 'dsr01'
DR_init.__dsr__model = 'e0509'
rclpy.init()
node = rclpy.create_node('llm_robot', namespace='dsr01')
DR_init.__dsr__node = node
from DSR_ROBOT2 import movel, movej, posx, posj, \
    set_velx, set_accx, set_velj, set_accj, \
    ROBOT_MODE_AUTONOMOUS, set_robot_mode, DR_BASE, \
    get_current_posx

set_robot_mode(ROBOT_MODE_AUTONOMOUS)
set_velj(10); set_accj(20)
set_velx(10, 5); set_accx(20, 10)

# ── 캘리브레이션 로드 ─────────────────────────
with open("calibration.json") as f:
    _calib = json.load(f)

# Perspective Transform 행렬
M = np.float32(_calib["transform_matrix"])

# 테이블 높이 (캘리브레이션에서 읽음)
TABLE_Z = _calib["table_z"]

print(f"✅ 캘리브레이션 로드 완료")
print(f"   TABLE_Z: {TABLE_Z:.1f}mm")

# ── 상수 ──────────────────────────────────────
MAX_ATTEMPT  = 10
MAX_ALIGN    = 5
ERR_THRESH   = 20    # 허용 오차 (px)
KP           = 0.6   # P-제어 게인

BIN_COORDS = {
    "bottle": (500,  300, TABLE_Z),
    "can":    (500, -300, TABLE_Z),
    "paper":  (400,    0, TABLE_Z),
    "general":(-500,   0, TABLE_Z),
}

# ── ROI 로드 ──────────────────────────────────
with open("roi_config.json") as f:
    _roi = json.load(f)["roi"]
ROI = (_roi["x1"], _roi["y1"], _roi["x2"], _roi["y2"])
print(f"✅ ROI: {ROI}")

# ── Perspective Transform 좌표 변환 ───────────
def pixel_to_robot(px: int, py: int):
    """
    픽셀 좌표 → 로봇 좌표 (Perspective Transform)
    캘리브레이션 행렬 사용 → x,y 둘 다 정확
    """
    pt  = np.float32([[[float(px), float(py)]]])
    res = cv2.perspectiveTransform(pt, M)
    robot_x = float(res[0][0][0])
    robot_y = float(res[0][0][1])
    return robot_x, robot_y, TABLE_Z

def robot_to_pixel(robot_x: float, robot_y: float):
    """
    로봇 좌표 → 픽셀 좌표 (역변환)
    정렬 확인용
    """
    M_inv = cv2.invert(M)[1]
    pt    = np.float32([[[robot_x, robot_y]]])
    res   = cv2.perspectiveTransform(pt, M_inv)
    px = int(res[0][0][0])
    py = int(res[0][0][1])
    return px, py

def get_tcp():
    """현재 TCP 좌표"""
    try:
        pos = get_current_posx(DR_BASE)
        return pos[0][0], pos[0][1], pos[0][2]
    except:
        return 0.0, 0.0, 0.0

# ── 그리퍼 ───────────────────────────────────
def gripper_open():
    print("    [그리퍼 열기]")
    subprocess.run(['ros2','service','call','/dsr01/gripper/open',
                    'std_srvs/srv/Trigger','{}'], capture_output=True)

def gripper_close():
    print("    [그리퍼 닫기]")
    subprocess.run(['ros2','service','call','/dsr01/gripper/close',
                    'std_srvs/srv/Trigger','{}'], capture_output=True)

# ── RViz 마커 ─────────────────────────────────
qos = QoSProfile(
    depth=10,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE
)
marker_pub = node.create_publisher(Marker, '/visualization_marker', qos)

def _make_marker(ns, mid, mtype, x, y, z, sx, sy, sz, r, g, b, a=0.9, text=""):
    m = Marker()
    m.header.frame_id = "base_link"
    m.header.stamp = node.get_clock().now().to_msg()
    m.ns = ns; m.id = mid
    m.action = Marker.ADD; m.type = mtype
    m.pose.position.x = x; m.pose.position.y = y; m.pose.position.z = z
    m.pose.orientation.w = 1.0
    m.scale.x = sx; m.scale.y = sy; m.scale.z = sz
    m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = a
    if text: m.text = text
    return m

def publish_object_marker(rx, ry, rz, obj_type, visible=True):
    color_map = {
        "bottle":(0.2,0.5,1.0), "can":(1.0,0.2,0.2),
        "paper": (1.0,1.0,0.3), "general":(0.5,0.5,0.5),
    }
    r, g, b = color_map.get(obj_type, (1.0,1.0,1.0))
    m = _make_marker("objects", 0, Marker.CYLINDER,
                     rx/1000, ry/1000, rz/1000, 0.07, 0.07, 0.20, r, g, b)
    if not visible: m.action = Marker.DELETE
    marker_pub.publish(m)
    rclpy.spin_once(node, timeout_sec=0.05)

def publish_gripper_marker(rx, ry, rz):
    m = _make_marker("gripper", 99, Marker.SPHERE,
                     rx/1000, ry/1000, rz/1000,
                     0.05, 0.05, 0.05, 0.0, 1.0, 0.0)
    marker_pub.publish(m)
    rclpy.spin_once(node, timeout_sec=0.05)

def publish_bin_markers():
    bins = [
        (10,"bottle", 0.5, 0.3,0.05, 0.2,0.5,1.0),
        (11,"can",    0.5,-0.3,0.05, 1.0,0.2,0.2),
        (12,"paper",  0.4, 0.0,0.05, 1.0,1.0,0.3),
        (13,"general",-0.5,0.0,0.05, 0.5,0.5,0.5),
    ]
    for (mid, name, x, y, z, r, g, b) in bins:
        marker_pub.publish(
            _make_marker("bins", mid, Marker.CUBE,
                         x, y, z, 0.15, 0.15, 0.10, r, g, b, 0.5))
        marker_pub.publish(
            _make_marker("bin_labels", mid+10, Marker.TEXT_VIEW_FACING,
                         x, y, z+0.12, 0.01, 0.01, 0.07,
                         1.0, 1.0, 1.0, 1.0, text=name))
        rclpy.spin_once(node, timeout_sec=0.05)

# ── Claude API ───────────────────────────────
client = anthropic.Anthropic()

def ask_claude(image: np.ndarray) -> dict:
    """ROI 크롭 → Claude sonnet-4-5 → 물체 픽셀 좌표"""
    x1, y1, x2, y2 = ROI
    roi_img = image[y1:y2, x1:x2]
    h, w    = roi_img.shape[:2]

    cv2.imwrite("last_roi.jpg", roi_img)

    _, buf = cv2.imencode('.jpg', roi_img)
    b64    = base64.b64encode(buf).decode('utf-8')

    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64
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
}}"""}
            ]
        }]
    )

    text = resp.content[0].text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    result = json.loads(text.strip())

    # 크롭 기준 → 전체 이미지 픽셀 변환
    if result.get("found"):
        result["pixel_x"] += x1
        result["pixel_y"] += y1

    return result

def save_debug(image, plan, gripper_px=None, gripper_py=None,
               err_x=0, err_y=0, tag="debug"):
    img = image.copy()
    x1, y1, x2, y2 = ROI
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),2)
    cv2.putText(img,"ROI",(x1+5,y1+20),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

    if plan and plan.get("found"):
        px = plan["pixel_x"]
        py = plan["pixel_y"]
        cv2.circle(img,(px,py),18,(0,0,255),3)
        cv2.drawMarker(img,(px,py),(0,0,255),cv2.MARKER_CROSS,35,3)
        cv2.putText(img, plan["object"],(px+18,py),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

        if gripper_px is not None:
            cv2.circle(img,(gripper_px,gripper_py),12,(0,220,0),3)
            cv2.drawMarker(img,(gripper_px,gripper_py),
                           (0,220,0),cv2.MARKER_CROSS,35,3)
            cv2.line(img,(gripper_px,gripper_py),(px,py),(0,200,0),2)
            cv2.putText(img,
                        f"err:({err_x:+d},{err_y:+d})px",
                        (10,35),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,200),2)

        cv2.putText(img,f"obj:{plan['object']} px:({px},{py})",
                    (5,65),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,50,255),2)

    cv2.imwrite(f"{tag}.jpg", img)
    print(f"  📸 {tag}.jpg 저장")

# ── 메인 로직 ─────────────────────────────────
def run(cam: CameraNode):

    print(f"\n{'='*55}")
    print(f"🤖 분리수거 에이전트 (Perspective Transform)")
    print(f"   모델: claude-sonnet-4-5")
    print(f"   최대 시도: {MAX_ATTEMPT}회")
    print(f"{'='*55}\n")

    movej(posj(0, 0, 90, 0, 90, 0))
    time.sleep(2)
    gripper_open()

    hx, hy, hz = get_tcp()
    print(f"🏠 홈 TCP: ({hx:.1f}, {hy:.1f}, {hz:.1f})mm\n")

    obj_type = None
    cur_x    = 0.0
    cur_y    = 0.0

    for attempt in range(1, MAX_ATTEMPT + 1):

        print(f"\n{'─'*50}")
        print(f"[시도 {attempt}/{MAX_ATTEMPT}]")
        print(f"{'─'*50}")

        # ── 1. 카메라 + Claude 감지 ───────────
        color, _, _ = cam.get_frames()
        print("🧠 Claude 감지 중...")
        plan = ask_claude(color)
        print(f"  결과: {json.dumps(plan, ensure_ascii=False)}")
        save_debug(color, plan, tag=f"detect_{attempt}")

        if not plan.get("found"):
            if attempt == 1:
                print("❌ 물체 없음. 종료.")
                return
            else:
                print("✅ 물체 사라짐 → 집기 성공!")
                break

        obj_type = plan["object"]
        px       = plan["pixel_x"]
        py       = plan["pixel_y"]

        # ── 2. Perspective Transform 좌표 변환 ─
        cur_x, cur_y, cur_z = pixel_to_robot(px, py)
        print(f"  픽셀({px},{py}) → 로봇({cur_x:.1f},{cur_y:.1f},{cur_z:.1f})mm")

        publish_object_marker(cur_x, cur_y, cur_z, obj_type)

        # ── 3. Approach ───────────────────────
        approach_z = cur_z + 50
        print(f"🦾 Approach → ({cur_x:.1f}, {cur_y:.1f}, {approach_z:.1f})")
        movel(posx(cur_x, cur_y, approach_z, 0, 180, 0), ref=DR_BASE)
        time.sleep(1.5)

        ax, ay, az = get_tcp()
        print(f"  실제 TCP: ({ax:.1f}, {ay:.1f}, {az:.1f})mm")
        publish_gripper_marker(ax, ay, az)

        # ── 4. P-제어 정렬 루프 ───────────────
        print("📷 위치 정렬 확인 중...")
        aligned = False

        for align in range(1, MAX_ALIGN + 1):

            check_color, _, _ = cam.get_frames()
            check = ask_claude(check_color)

            if not check.get("found"):
                print(f"  [{align}] 물체 감지 실패")
                continue

            obj_px_now = check["pixel_x"]
            obj_py_now = check["pixel_y"]

            # 현재 그리퍼 픽셀 (역변환)
            gripper_px, gripper_py = robot_to_pixel(cur_x, cur_y)

            # x, y 픽셀 오차 둘 다 계산
            err_x = obj_px_now - gripper_px
            err_y = obj_py_now - gripper_py

            rx, ry, rz = get_tcp()
            print(f"  [{align}] 물체픽셀:({obj_px_now},{obj_py_now}) "
                  f"그리퍼픽셀:({gripper_px},{gripper_py}) "
                  f"오차:({err_x:+d},{err_y:+d})px")
            print(f"         실제TCP:({rx:.1f},{ry:.1f})")

            save_debug(check_color, check,
                       gripper_px, gripper_py,
                       err_x, err_y,
                       tag=f"align_{attempt}_{align}")
            publish_gripper_marker(rx, ry, rz)

            if abs(err_x) <= ERR_THRESH and abs(err_y) <= ERR_THRESH:
                print(f"  ✅ 정렬 완료!")
                aligned = True
                break

            # P-제어: 픽셀 오차 → 로봇 좌표 보정
            # 보정 후 새 물체 픽셀 위치로 다시 변환
            new_px = obj_px_now
            new_py = obj_py_now
            new_x, new_y, _ = pixel_to_robot(new_px, new_py)

            # P-제어 게인 적용
            cur_x += (new_x - cur_x) * KP
            cur_y += (new_y - cur_y) * KP

            print(f"  🔧 P보정 → 새 로봇({cur_x:.1f},{cur_y:.1f})")

            movel(posx(cur_x, cur_y, approach_z, 0, 180, 0), ref=DR_BASE)
            time.sleep(1.2)

            bx, by, bz = get_tcp()
            print(f"  보정 후 실제TCP:({bx:.1f},{by:.1f})")
            publish_gripper_marker(bx, by, bz)

        if not aligned:
            print(f"  ⚠️ 정렬 미완 → 현재 위치로 집기")

        # ── 5. 집기 ───────────────────────────
        print(f"🦾 하강 → z={cur_z:.1f}mm")
        movel(posx(cur_x, cur_y, cur_z, 0, 180, 0), ref=DR_BASE)
        time.sleep(1.2)

        gx, gy, gz = get_tcp()
        print(f"  집기 직전 TCP: ({gx:.1f}, {gy:.1f}, {gz:.1f})mm")

        gripper_close()
        time.sleep(0.6)

        # ── 6. Lift ───────────────────────────
        lift_z = cur_z + 150
        print(f"🦾 Lift → z={lift_z:.1f}mm")
        movel(posx(cur_x, cur_y, lift_z, 0, 180, 0), ref=DR_BASE)
        time.sleep(1.5)

        # ── 7. 집기 성공 확인 ─────────────────
        print("📷 집기 확인 중...")
        final_color, _, _ = cam.get_frames()
        final = ask_claude(final_color)
        save_debug(final_color, final, tag=f"check_{attempt}")

        if not final.get("found"):
            print("✅ 물체 사라짐 → 집기 성공!")
            break

        # 같은 위치 물체인지 확인
        vx, vy = final["pixel_x"], final["pixel_y"]
        dist = np.sqrt((vx-px)**2 + (vy-py)**2)
        if dist > 60:
            print(f"✅ 다른 위치 물체 (거리:{dist:.0f}px) → 성공!")
            break

        print(f"❌ 같은 위치 물체 여전히 있음 → 재시도")
        gripper_open()
        time.sleep(0.3)
        movel(posx(cur_x, cur_y, lift_z, 0, 180, 0), ref=DR_BASE)
        time.sleep(1.0)

    else:
        print(f"\n🚫 {MAX_ATTEMPT}회 모두 실패. 종료.")
        movej(posj(0, 0, 0, 0, 0, 0))
        return

    # ── 8. 분리수거함 ─────────────────────────
    if obj_type is None:
        obj_type = "general"

    bx, by, bz = BIN_COORDS[obj_type]
    print(f"\n♻️  {obj_type} → 분리수거함 ({bx}, {by})")
    movel(posx(bx, by, bz+150, 0, 180, 0), ref=DR_BASE)
    time.sleep(2)
    gripper_open()
    time.sleep(0.5)

    # ── 9. 완료 ───────────────────────────────
    publish_object_marker(0, 0, 0, obj_type, visible=False)
    movej(posj(0, 0, 0, 0, 0, 0))
    time.sleep(2)

    print(f"\n{'='*55}")
    print(f"✅ 분리수거 완료!")
    print(f"   물체    : {obj_type}")
    print(f"   최종 좌표: ({cur_x:.1f}, {cur_y:.1f})mm")
    print(f"{'='*55}")

# ── 메인 ──────────────────────────────────────
if __name__ == "__main__":

    cam = CameraNode()

    print("📍 RViz 마커 퍼블리시...")
    for _ in range(15):
        publish_bin_markers()
        time.sleep(0.2)
    print("✅ 마커 완료\n")

    try:
        run(cam)
    finally:
        cam.release()
        node.destroy_node()
        rclpy.shutdown()
        print("종료!")