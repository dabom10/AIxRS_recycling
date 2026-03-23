#!/usr/bin/env python3
"""
wrist_explore_node.py
물체를 집은 상태에서 손목 관절(j4, j5, j6)로 다양하게 탐색하는 노드
RViz 시뮬레이션 가능 (virtual 모드)

실행:
  source ~/doosan_gripper_ws/install/setup.bash
  export ROS_DOMAIN_ID=29
  cd ~/workspace/robot_llm
  source venv/bin/activate
  python wrist_explore_node.py
"""

import rclpy
import DR_init
import time
import math
import threading
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

# ── ROS2 + DSR 초기화 ──────────────────────────────────────────
ROBOT_ID    = "dsr01"
ROBOT_MODEL = "e0509"
DR_init.__dsr__id    = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
node = rclpy.create_node('wrist_explore_node', namespace=ROBOT_ID)
DR_init.__dsr__node = node

from DSR_ROBOT2 import (
    movej, posj,
    set_velj, set_accj,
    set_robot_mode, get_current_posj,
    ROBOT_MODE_AUTONOMOUS
)

set_robot_mode(ROBOT_MODE_AUTONOMOUS)
set_velj(15)
set_accj(30)

print("✅ wrist_explore_node 시작")

# ── QoS 설정 ───────────────────────────────────────────────────
qos = QoSProfile(
    depth=10,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE
)

# ── 탐색 패턴 정의 ─────────────────────────────────────────────
# 각 패턴: (j4, j5, j6) 오프셋 (degree)
# 현재 j1, j2, j3는 유지하고 손목만 변경

EXPLORE_PATTERNS = {
    "neutral": {
        "desc": "중립 자세",
        "wrist": (0, 0, 0),
        "color": (0.5, 0.5, 0.5)
    },
    "tilt_left": {
        "desc": "왼쪽으로 기울이기",
        "wrist": (30, 0, 0),
        "color": (0.2, 0.6, 1.0)
    },
    "tilt_right": {
        "desc": "오른쪽으로 기울이기",
        "wrist": (-30, 0, 0),
        "color": (1.0, 0.4, 0.2)
    },
    "tilt_forward": {
        "desc": "앞으로 기울이기",
        "wrist": (0, 30, 0),
        "color": (0.2, 1.0, 0.4)
    },
    "tilt_backward": {
        "desc": "뒤로 기울이기",
        "wrist": (0, -30, 0),
        "color": (1.0, 1.0, 0.2)
    },
    "rotate_cw": {
        "desc": "시계방향 회전",
        "wrist": (0, 0, 45),
        "color": (0.8, 0.2, 0.8)
    },
    "rotate_ccw": {
        "desc": "반시계방향 회전",
        "wrist": (0, 0, -45),
        "color": (0.2, 0.8, 0.8)
    },
    "inspect_top": {
        "desc": "위에서 내려다보기",
        "wrist": (0, -45, 0),
        "color": (1.0, 0.6, 0.0)
    },
    "inspect_side": {
        "desc": "옆면 보기",
        "wrist": (45, 0, 90),
        "color": (0.0, 0.8, 0.6)
    },
    "sweep_scan": {
        "desc": "스윕 스캔 (j6 회전)",
        "wrist": (0, 0, 0),
        "color": (1.0, 0.2, 0.5),
        "sweep": True  # 스윕 동작 플래그
    },
}

# ── 현재 상태 ──────────────────────────────────────────────────
current_pattern = "neutral"
base_joints = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]  # 집은 상태 기본 자세
is_executing = False

# ── Publisher 설정 ─────────────────────────────────────────────
marker_pub = node.create_publisher(MarkerArray, '/wrist_explore/markers', qos)
status_pub = node.create_publisher(String, '/wrist_explore/status', 10)
cmd_sub = node.create_subscription(
    String, '/wrist_explore/command',
    lambda msg: execute_pattern(msg.data),
    10
)

# ── 마커 퍼블리시 ─────────────────────────────────────────────
def publish_pattern_markers():
    """RViz에 탐색 패턴들을 마커로 표시"""
    array = MarkerArray()
    
    # 패턴 목록 표시 (텍스트 마커)
    for i, (name, info) in enumerate(EXPLORE_PATTERNS.items()):
        # 패턴 라벨
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = node.get_clock().now().to_msg()
        m.ns = "patterns"
        m.id = i
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        
        # 원형으로 배치
        angle = (2 * math.pi * i) / len(EXPLORE_PATTERNS)
        radius = 0.4
        m.pose.position.x = 0.3 + radius * math.cos(angle)
        m.pose.position.y = radius * math.sin(angle)
        m.pose.position.z = 1.2
        m.pose.orientation.w = 1.0
        
        m.scale.z = 0.04
        r, g, b = info["color"]
        
        # 현재 실행 중인 패턴 강조
        if name == current_pattern:
            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 1.0
            m.scale.z = 0.06
        else:
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 0.7
        
        m.text = f"{name}\n{info['desc']}"
        array.markers.append(m)
        
        # 패턴 구 마커
        sphere = Marker()
        sphere.header.frame_id = "base_link"
        sphere.header.stamp = node.get_clock().now().to_msg()
        sphere.ns = "pattern_spheres"
        sphere.id = i + 100
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = 0.3 + radius * math.cos(angle)
        sphere.pose.position.y = radius * math.sin(angle)
        sphere.pose.position.z = 1.1
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = 0.05
        sphere.scale.y = 0.05
        sphere.scale.z = 0.05
        sphere.color.r = r
        sphere.color.g = g
        sphere.color.b = b
        sphere.color.a = 0.9 if name == current_pattern else 0.4
        array.markers.append(sphere)
    
    # 현재 패턴 화살표
    arrow = Marker()
    arrow.header.frame_id = "base_link"
    arrow.header.stamp = node.get_clock().now().to_msg()
    arrow.ns = "current_arrow"
    arrow.id = 200
    arrow.type = Marker.ARROW
    arrow.action = Marker.ADD
    arrow.pose.position.x = 0.3
    arrow.pose.position.y = 0.0
    arrow.pose.position.z = 1.0
    arrow.pose.orientation.w = 1.0
    arrow.scale.x = 0.2
    arrow.scale.y = 0.03
    arrow.scale.z = 0.03
    arrow.color.r = 1.0
    arrow.color.g = 1.0
    arrow.color.b = 0.0
    arrow.color.a = 0.8
    array.markers.append(arrow)
    
    # 물체 표시 (집고 있는 물체)
    obj = Marker()
    obj.header.frame_id = "link_6"  # 그리퍼 끝에 붙어있음
    obj.header.stamp = node.get_clock().now().to_msg()
    obj.ns = "held_object"
    obj.id = 300
    obj.type = Marker.CYLINDER
    obj.action = Marker.ADD
    obj.pose.position.x = 0.0
    obj.pose.position.y = 0.0
    obj.pose.position.z = 0.12  # 그리퍼 아래
    obj.pose.orientation.w = 1.0
    obj.scale.x = 0.06
    obj.scale.y = 0.06
    obj.scale.z = 0.15
    obj.color.r = 0.2
    obj.color.g = 0.6
    obj.color.b = 1.0
    obj.color.a = 0.8
    array.markers.append(obj)
    
    marker_pub.publish(array)

# ── 패턴 실행 ─────────────────────────────────────────────────
def execute_pattern(pattern_name: str):
    global current_pattern, is_executing, base_joints

    if is_executing:
        print(f"⚠️  실행 중 - 무시: {pattern_name}")
        return

    if pattern_name not in EXPLORE_PATTERNS:
        print(f"❌ 알 수 없는 패턴: {pattern_name}")
        print(f"   가능한 패턴: {list(EXPLORE_PATTERNS.keys())}")
        return

    is_executing = True
    current_pattern = pattern_name
    info = EXPLORE_PATTERNS[pattern_name]

    print(f"\n🔍 탐색 패턴: {pattern_name} - {info['desc']}")

    # 현재 관절 읽기
    try:
        current = get_current_posj()
        j = [float(current[i]) for i in range(6)]
        # j1, j2, j3는 유지, j4~j6만 오프셋 적용
        base_joints = [j[0], j[1], j[2],
                      j[3], j[4], j[5]]
    except Exception as e:
        print(f"⚠️  관절 읽기 실패, 기본값 사용: {e}")

    dj4, dj5, dj6 = info["wrist"]

    # 스윕 스캔인 경우
    if info.get("sweep"):
        def sweep():
            for angle in range(-60, 61, 15):
                target = posj(
                    base_joints[0],
                    base_joints[1],
                    base_joints[2],
                    base_joints[3] + dj4,
                    base_joints[4] + dj5,
                    base_joints[5] + angle
                )
                movej(target, vel=20, acc=40)
                publish_pattern_markers()
                
                # 상태 퍼블리시
                msg = String()
                msg.data = f"sweep_scan|j6={angle}"
                status_pub.publish(msg)
                time.sleep(0.3)
            
            # 원래 위치로
            home = posj(*base_joints)
            movej(home, vel=15, acc=30)
            finish()

        threading.Thread(target=sweep, daemon=True).start()
        return

    # 일반 패턴 실행
    target = posj(
        base_joints[0],
        base_joints[1],
        base_joints[2],
        base_joints[3] + dj4,
        base_joints[4] + dj5,
        base_joints[5] + dj6
    )
    movej(target, vel=15, acc=30)
    
    # 상태 퍼블리시
    msg = String()
    msg.data = f"{pattern_name}|{info['desc']}"
    status_pub.publish(msg)
    
    print(f"✅ 완료: {pattern_name}")
    print(f"   j4={base_joints[3]+dj4:.1f} "
          f"j5={base_joints[4]+dj5:.1f} "
          f"j6={base_joints[5]+dj6:.1f}")
    
    finish()

def finish():
    global is_executing
    is_executing = False
    publish_pattern_markers()

# ── 자동 순회 ─────────────────────────────────────────────────
def auto_explore(interval=3.0):
    """모든 패턴을 자동으로 순회"""
    print("\n🔄 자동 탐색 시작 (Ctrl+C로 중지)")
    patterns = list(EXPLORE_PATTERNS.keys())
    
    for name in patterns:
        execute_pattern(name)
        time.sleep(interval)
        
        # neutral로 복귀
        execute_pattern("neutral")
        time.sleep(1.0)
    
    print("✅ 자동 탐색 완료")

# ── 마커 타이머 ───────────────────────────────────────────────
timer = node.create_timer(0.5, publish_pattern_markers)

# ── 메인 ──────────────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("🤖 손목 탐색 노드 (wrist_explore_node)")
    print("="*55)
    print("명령:")
    print("  1. 단일 패턴 실행")
    for name, info in EXPLORE_PATTERNS.items():
        print(f"     {name:20s} → {info['desc']}")
    print("  2. auto  → 모든 패턴 자동 순회")
    print("  3. quit  → 종료")
    print("\nROS2 토픽으로도 제어 가능:")
    print("  ros2 topic pub /dsr01/wrist_explore/command "
          "std_msgs/msg/String \"{data: 'tilt_left'}\"")
    print()

    # 초기 자세: 물체 집은 상태
    print("📍 초기 자세 이동 (물체 집은 상태)...")
    movej(posj(*base_joints), vel=20, acc=40)
    publish_pattern_markers()

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    spin_thread = threading.Thread(
        target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            cmd = input("\n명령: ").strip()
            
            if cmd.lower() in ['quit', 'exit', '종료']:
                break
            elif cmd.lower() == 'auto':
                threading.Thread(
                    target=auto_explore, daemon=True).start()
            elif cmd in EXPLORE_PATTERNS:
                execute_pattern(cmd)
            else:
                print(f"❌ 알 수 없는 명령: {cmd}")
                print(f"   가능: {list(EXPLORE_PATTERNS.keys()) + ['auto']}")
    
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("\n종료!")

if __name__ == "__main__":
    main()