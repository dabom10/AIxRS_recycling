import rclpy
import DR_init
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import Int32, Bool, String, Float32

# ── 로봇 설정 ──────────────────────────────────────────────
ROBOT_ID    = "dsr01"
ROBOT_MODEL = "e0509"
VELOCITY    = 60
ACC         = 60

# ── 그리퍼 설정 ───────────────────────────────────────────
GRIPPER_CLOSE_VAL = 600
GRIPPER_OPEN_SVC  = f"/{ROBOT_ID}/gripper/open"
GRIPPER_POS_TOPIC = f"/{ROBOT_ID}/gripper/position_cmd"

# ── 관절 위치 ─────────────────────────────────────────────
HOME_J     = [  0.0,   0.0,  90.0,   0.0,  90.0,   0.0]
APPROACH_J = [-23.501, 20.29, 66.166, -0.002, 93.56, -23.49]

# ── 작업 위치 ─────────────────────────────────────────────
GRASP_X   = [460.009,   -0.024, 404.942, 179.901, 179.991, 179.903]
PLASTIC_X = [459.982, -250.000, 404.905, 164.881, 179.983, 164.891]
PAPER_X   = [269.979, -250.036, 404.880, 169.404, 179.981, 169.421]

LIFT_Z = 500.0
Z_IDX  = 2


def _modified(base: list, idx: int, val: float) -> list:
    pos = list(base)
    pos[idx] = val
    return pos


def camera_to_robot(cx, cy, depth):
    """카메라 좌표 → 로봇 좌표 변환 (TODO: 캘리브레이션 행렬로 교체)"""
    rx, ry, rz = GRASP_X[3], GRASP_X[4], GRASP_X[5]
    return [cx, cy, depth, rx, ry, rz]


class MoveNode(Node):
    def __init__(self):
        super().__init__("recycle_move", namespace=ROBOT_ID)

        # DSR API — main()에서 주입
        self.movej:   callable = None
        self.movel:   callable = None
        self.DR_BASE: int      = None
        self.posj:    callable = None
        self.posx:    callable = None

        # 그리퍼
        self._open_cli = self.create_client(Trigger, GRIPPER_OPEN_SVC)
        self._pos_pub  = self.create_publisher(Int32, GRIPPER_POS_TOPIC, 10)

        # done 퍼블리셔
        self._done_pub = self.create_publisher(Bool, "/reccycle1/done", 10)

        # 객체인식 구독
        self._class_name = self._center_x = self._center_y = self._depth = None

        self.create_subscription(String,  "/recycle1/class_name", lambda m: setattr(self, "_class_name", m.data), 10)
        self.create_subscription(Float32, "/recycle1/center_x",   lambda m: setattr(self, "_center_x",   m.data), 10)
        self.create_subscription(Float32, "/recycle1/center_y",   lambda m: setattr(self, "_center_y",   m.data), 10)
        self.create_subscription(Float32, "/recycle1/depth",      self._cb_depth, 10)

    # ── 구독 콜백 ─────────────────────────────────────────
    def _cb_depth(self, msg):
        self._depth = msg.data
        if all(v is not None for v in [self._class_name, self._center_x, self._center_y]):
            self.run_cycle(self._class_name, self._center_x, self._center_y, self._depth)
            self._class_name = self._center_x = self._center_y = self._depth = None

    # ── 이동 헬퍼 ─────────────────────────────────────────
    def _movej(self, joint: list):
        self.movej(self.posj(*joint), vel=VELOCITY, acc=ACC)

    def _movel(self, task: list):
        self.movel(self.posx(*task), vel=[50, 50], acc=[100, 100], ref=self.DR_BASE)

    # ── 그리퍼 ───────────────────────────────────────────
    def gripper_open(self):
        self._open_cli.wait_for_service()
        rclpy.spin_until_future_complete(self, self._open_cli.call_async(Trigger.Request()))

    def gripper_close(self):
        msg = Int32()
        msg.data = GRIPPER_CLOSE_VAL
        self._pos_pub.publish(msg)

    # ── 동작 함수 ─────────────────────────────────────────
    def go_home(self):
        self._movej(HOME_J)

    def grasp_position(self, grasp_x):
        self._movej(APPROACH_J)
        self._movel(grasp_x)
        self.gripper_close()

    def throw_object_plastic(self, grasp_x):
        self._movel(_modified(grasp_x, Z_IDX, LIFT_Z))
        self._movej(APPROACH_J)
        self._movel(PLASTIC_X)
        self.gripper_open()

    def throw_object_paper(self, grasp_x):
        self._movel(_modified(grasp_x, Z_IDX, LIFT_Z))
        self._movej(APPROACH_J)
        self._movel(PAPER_X)
        self.gripper_open()

    # ── 한 사이클 ─────────────────────────────────────────
    def run_cycle(self, class_name, cx, cy, depth):
        grasp_x = camera_to_robot(cx, cy, depth)

        self.go_home()
        self.grasp_position(grasp_x)

        if class_name == "plastic":
            self.throw_object_plastic(grasp_x)
        else:
            self.throw_object_paper(grasp_x)

        self.go_home()
        self._done_pub.publish(Bool(data=True))


def main():
    rclpy.init()
    DR_init.__dsr__id    = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL

    node = MoveNode()
    DR_init.__dsr__node = node

    from DSR_ROBOT2 import movej, movel, DR_BASE
    from DR_common2 import posj, posx

    node.movej   = movej
    node.movel   = movel
    node.DR_BASE = DR_BASE
    node.posj    = posj
    node.posx    = posx

    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()