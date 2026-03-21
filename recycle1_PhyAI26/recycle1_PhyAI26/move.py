# 두산 로봇팔 E0509와 그리퍼 RH-P12-RN-A 제어 코드

import rclpy
import DR_init

# ── 로봇 설정 ──────────────────────────────────────────────
ROBOT_ID    = "dsr01"
ROBOT_MODEL = "e0509"
VELOCITY    = 60
ACC         = 60
TOLERANCE   = 0.5  # 도착 판정 오차 허용값 (deg)

# ── 관절 위치 (posj) ── [J1, J2, J3, J4, J5, J6] deg ─────
HOME_J      = [  0.0,    0.0,   90.0,   0.0,    90.0,    0.0]      # 호밍 (안전 대기 자세)
APPROACH_J  = [-23.501,  20.29, 66.166, -0.002,  93.56, -23.49]   # 파지·투척 공통 관절 접근

# ── 작업 위치 (posx) ── [X, Y, Z, Rx, Ry, Rz] mm/deg ─────
GRASP_X     = [460.009,   -0.024, 404.942, 179.901, 179.991, 179.903]  # 파지 대기
PLASTIC_X   = [459.982, -250.000, 404.905, 164.881, 179.983, 164.891]  # 플라스틱 투척
PAPER_X     = [269.979, -250.036, 404.880, 169.404, 179.981, 169.421]  # 종이 투척

# ── 하강/상승 보조 설정 ────────────────────────────────────
LIFT_Z      = 500.0   # 이동 전 상승 높이 (mm)
Z_IDX       = 2       # posx 배열에서 Z 인덱스


# ── 헬퍼 ──────────────────────────────────────────────────
def _modified(base: list, idx: int, val: float) -> list:
    """base 좌표 배열에서 idx 번 인덱스 값만 val로 교체한 새 리스트 반환"""
    pos = list(base)
    pos[idx] = val
    return pos


# ── 메인 클래스 ───────────────────────────────────────────
class RecycleMove:
    def __init__(self, node, movej, movel, mwait, get_current_posj, posj, posx, DR_BASE):
        self._node            = node
        self.movej            = movej
        self.movel            = movel
        self.mwait            = mwait
        self.get_current_posj = get_current_posj
        self.posj             = posj
        self.posx             = posx
        self.DR_BASE          = DR_BASE

    # ── 내부 이동 헬퍼 ────────────────────────────────────
    def _movej(self, joint: list):
        self.movej(self.posj(*joint), vel=VELOCITY, acc=ACC)
        self.mwait()

    def _movel(self, task: list):
        self.movel(self.posx(*task), vel=[50, 50], acc=[100, 100], ref=self.DR_BASE)
        self.mwait()

    # ── 공개 메서드 ───────────────────────────────────────
    def go_home(self):
        """호밍 위치로 이동 (안전 대기 자세)"""
        self._movej(HOME_J)

    def grasp_position(self):
        """파지 대기 위치로 이동 (movej → movel 하강)"""
        self._movej(APPROACH_J)
        self._movel(GRASP_X)

    def throw_object_plastic(self):
        """플라스틱 분리수거 위치로 이동 (상승 → movej → movel 하강)"""
        self._movel(_modified(GRASP_X, Z_IDX, LIFT_Z))
        self._movej(APPROACH_J)
        self._movel(PLASTIC_X)

    def throw_object_paper(self):
        """종이 분리수거 위치로 이동 (상승 → movej → movel 하강)"""
        self._movel(_modified(GRASP_X, Z_IDX, LIFT_Z))
        self._movej(APPROACH_J)
        self._movel(PAPER_X)

    def check_arrived(self, target_j: list) -> bool:
        """현재 관절 위치와 목표 위치 비교로 도착 확인"""
        current = self.get_current_posj()
        errors  = [abs(current[i] - target_j[i]) for i in range(6)]
        arrived = all(e < TOLERANCE for e in errors)
        if not arrived:
            self._node.get_logger().warn(f"위치 오차: {[round(e, 3) for e in errors]}")
        return arrived


def main():
    rclpy.init()

    # 1. DR_init 설정
    DR_init.__dsr__id    = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL

    # 2. 노드 생성 후 DR_init에 등록
    node = rclpy.create_node("recycle_move", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    # 3. 노드가 준비된 뒤 DSR_ROBOT2 import (g_node 고정 시점 보장)
    from DSR_ROBOT2 import movej, movel, mwait, get_current_posj, DR_BASE
    from DR_common2 import posj, posx

    mover = RecycleMove(node, movej, movel, mwait, get_current_posj, posj, posx, DR_BASE)

    mover.go_home()
    if not mover.check_arrived(HOME_J):
        node.get_logger().error("호밍 위치 도착 실패")

    mover.grasp_position()
    if not mover.check_arrived(APPROACH_J):
        node.get_logger().error("파지 위치 도착 실패")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
