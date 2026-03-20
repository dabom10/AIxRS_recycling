import json
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import pyrealsense2 as rs


# =========================
# 설정
# =========================
@dataclass
class CameraConfig:
    color_width: int = 1280
    color_height: int = 720
    depth_width: int = 1280
    depth_height: int = 720
    fps: int = 30


@dataclass
class ROIConfig:
    # 네 작업대 화면 기준 대충 중앙 테이블 영역
    x1_ratio: float = 0.22
    y1_ratio: float = 0.28
    x2_ratio: float = 0.78
    y2_ratio: float = 0.93


@dataclass
class DepthConfig:
    min_depth_mm: int = 150
    max_depth_mm: int = 2000
    table_hist_bin_mm: int = 8
    object_lift_mm: int = 18
    close_kernel: int = 7
    open_kernel: int = 5
    min_component_area: int = 2000


@dataclass
class SelectConfig:
    center_weight: float = 2.0
    area_weight: float = 1.0
    edge_penalty: float = 0.5


CAMERA = CameraConfig()
ROI = ROIConfig()
DEPTH = DepthConfig()
SELECT = SelectConfig()


# =========================
# 유틸
# =========================
@dataclass
class ROIBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1


def make_roi(image: np.ndarray, cfg: ROIConfig) -> ROIBox:
    h, w = image.shape[:2]
    x1 = int(w * cfg.x1_ratio)
    y1 = int(h * cfg.y1_ratio)
    x2 = int(w * cfg.x2_ratio)
    y2 = int(h * cfg.y2_ratio)

    x1 = max(0, min(x1, w - 2))
    y1 = max(0, min(y1, h - 2))
    x2 = max(x1 + 1, min(x2, w - 1))
    y2 = max(y1 + 1, min(y2, h - 1))
    return ROIBox(x1, y1, x2, y2)


def crop_roi(color: np.ndarray, depth_mm: np.ndarray, roi: ROIBox):
    return (
        color[roi.y1:roi.y2, roi.x1:roi.x2].copy(),
        depth_mm[roi.y1:roi.y2, roi.x1:roi.x2].copy(),
    )


def estimate_table_depth_mm(depth_roi_mm: np.ndarray, cfg: DepthConfig) -> Optional[int]:
    valid = depth_roi_mm[
        (depth_roi_mm >= cfg.min_depth_mm) &
        (depth_roi_mm <= cfg.max_depth_mm)
    ]

    if valid.size < 100:
        return None

    valid = valid.astype(np.int32)
    bins = np.arange(cfg.min_depth_mm, cfg.max_depth_mm + cfg.table_hist_bin_mm, cfg.table_hist_bin_mm)
    hist, edges = np.histogram(valid, bins=bins)
    idx = int(np.argmax(hist))
    table_depth = int((edges[idx] + edges[idx + 1]) / 2)
    return table_depth


def build_object_mask(depth_roi_mm: np.ndarray, table_depth_mm: int, cfg: DepthConfig) -> np.ndarray:
    valid = (
        (depth_roi_mm >= cfg.min_depth_mm) &
        (depth_roi_mm <= cfg.max_depth_mm)
    )

    # 테이블보다 더 가까운 것만 객체로 본다
    obj = (
        valid &
        (depth_roi_mm < (table_depth_mm - cfg.object_lift_mm))
    ).astype(np.uint8) * 255

    if cfg.close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel))
        obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, k)

    if cfg.open_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_kernel, cfg.open_kernel))
        obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, k)

    return obj


def keep_large_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == i] = 255

    return out

def pick_best_component(mask: np.ndarray, cfg: SelectConfig) -> Optional[np.ndarray]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    h, w = mask.shape[:2]

    # ROI 정중앙이 아니라 "작업대 앞쪽 중앙"을 기준점으로 사용
    cx0 = w / 2.0
    cy0 = h * 0.78

    diag = math.hypot(w, h)

    best_idx = None
    best_score = -1e18

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[i]

        # 너무 작은 노이즈 제거
        if area < 800:
            continue

        # ROI 위쪽 물체 제거
        if cy < h * 0.55:
            continue

        dist = math.hypot(cx - cx0, cy - cy0) / max(diag, 1e-6)

        # 아래쪽일수록 가산점
        bottom_bias = cy / float(h)

        score = (
            cfg.center_weight * (1.0 - dist)
            + cfg.area_weight * (area / float(w * h))
            + 0.8 * bottom_bias
        )

        # ROI 경계에 붙은 건 패널티
        margin = 5
        if x <= margin or y <= margin or (x + ww) >= (w - margin) or (y + hh) >= (h - margin):
            score -= cfg.edge_penalty

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        return None

    out = np.zeros_like(mask)
    out[labels == best_idx] = 255
    return out


def largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def contour_centroid(cnt: np.ndarray) -> Optional[Tuple[float, float]]:
    m = cv2.moments(cnt)
    if abs(m["m00"]) < 1e-6:
        return None
    return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])


def contour_bbox(cnt: np.ndarray) -> Tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(cnt)
    return x, y, x + w, y + h


def contour_obb(cnt: np.ndarray) -> Dict[str, Any]:
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (rw, rh), angle = rect
    box = cv2.boxPoints(rect).astype(np.int32)
    return {
        "center": (float(cx), float(cy)),
        "size": (float(rw), float(rh)),
        "angle": float(angle),
        "box": box,
    }


def pca_major_axis_angle_deg(cnt: np.ndarray) -> float:
    pts = cnt.reshape(-1, 2).astype(np.float32)
    mean = pts.mean(axis=0)
    pts0 = pts - mean
    cov = np.cov(pts0.T)
    evals, evecs = np.linalg.eigh(cov)
    v = evecs[:, np.argmax(evals)]
    return math.degrees(math.atan2(float(v[1]), float(v[0]))) % 180.0


def raycast_to_edge(mask: np.ndarray, start_xy: Tuple[float, float], direction_xy: np.ndarray) -> Tuple[float, float]:
    h, w = mask.shape[:2]
    x, y = start_xy
    last = (x, y)

    for _ in range(max(h, w) * 2):
        x += float(direction_xy[0])
        y += float(direction_xy[1])

        xi = int(round(x))
        yi = int(round(y))

        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            break
        if mask[yi, xi] == 0:
            break
        last = (x, y)

    return last


def grasp_line_from_mask(mask: np.ndarray, centroid_xy: Tuple[float, float], major_angle_deg: float) -> Dict[str, Any]:
    grasp_angle_deg = (major_angle_deg + 90.0) % 180.0
    r = math.radians(grasp_angle_deg)
    d = np.array([math.cos(r), math.sin(r)], dtype=np.float32)

    p1 = raycast_to_edge(mask, centroid_xy, d)
    p2 = raycast_to_edge(mask, centroid_xy, -d)
    width_px = float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    return {
        "grasp_angle_deg": float(grasp_angle_deg),
        "line_xyxy": [float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])],
        "width_px": width_px,
    }


def safe_depth_mm(depth_roi_mm: np.ndarray, u: int, v: int, window: int = 3) -> Optional[int]:
    h, w = depth_roi_mm.shape[:2]
    x1 = max(0, u - window)
    y1 = max(0, v - window)
    x2 = min(w, u + window + 1)
    y2 = min(h, v + window + 1)

    patch = depth_roi_mm[y1:y2, x1:x2]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return int(np.median(valid))


def deproject_to_xyz(intrinsics: rs.intrinsics, u: float, v: float, depth_mm: int) -> list[float]:
    depth_m = depth_mm / 1000.0
    xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], float(depth_m))
    return [float(xyz[0]), float(xyz[1]), float(xyz[2])]


def analyze_scene(color: np.ndarray, depth_mm: np.ndarray, intrinsics: rs.intrinsics):
    roi = make_roi(color, ROI)
    color_roi, depth_roi = crop_roi(color, depth_mm, roi)

    table_depth = estimate_table_depth_mm(depth_roi, DEPTH)
    if table_depth is None:
        return None, color.copy()

    obj_mask = build_object_mask(depth_roi, table_depth, DEPTH)
    obj_mask = keep_large_components(obj_mask, DEPTH.min_component_area)
    selected = pick_best_component(obj_mask, SELECT)
    if selected is None:
        return None, color.copy()

    cnt = largest_contour(selected)
    if cnt is None:
        return None, color.copy()

    centroid = contour_centroid(cnt)
    if centroid is None:
        return None, color.copy()

    bbox = contour_bbox(cnt)
    obb = contour_obb(cnt)
    major_angle = pca_major_axis_angle_deg(cnt)
    grasp = grasp_line_from_mask(selected, centroid, major_angle)

    cu, cv = int(round(centroid[0])), int(round(centroid[1]))
    dmm = safe_depth_mm(depth_roi, cu, cv)
    xyz = None
    if dmm is not None and dmm > 0:
        xyz = deproject_to_xyz(intrinsics, cu + roi.x1, cv + roi.y1, dmm)

    result = {
        "roi_xyxy": [roi.x1, roi.y1, roi.x2, roi.y2],
        "table_depth_mm": table_depth,
        "bbox_xyxy": [bbox[0] + roi.x1, bbox[1] + roi.y1, bbox[2] + roi.x1, bbox[3] + roi.y1],
        "centroid_uv": [float(centroid[0] + roi.x1), float(centroid[1] + roi.y1)],
        "depth_at_centroid_mm": dmm,
        "centroid_xyz_m": xyz,
        "major_axis_angle_deg": float(major_angle),
        "grasp_angle_deg": float(grasp["grasp_angle_deg"]),
        "grasp_width_px": float(grasp["width_px"]),
        "grasp_line_xyxy": [
            float(grasp["line_xyxy"][0] + roi.x1),
            float(grasp["line_xyxy"][1] + roi.y1),
            float(grasp["line_xyxy"][2] + roi.x1),
            float(grasp["line_xyxy"][3] + roi.y1),
        ],
        "obb_angle_deg": float(obb["angle"]),
    }

    vis = color.copy()
    cv2.rectangle(vis, (roi.x1, roi.y1), (roi.x2, roi.y2), (100, 100, 100), 2)

    overlay = vis.copy()
    green = np.zeros((roi.h, roi.w, 3), dtype=np.uint8)
    green[:, :, 1] = (selected > 0).astype(np.uint8) * 180

    roi_img = overlay[roi.y1:roi.y2, roi.x1:roi.x2]
    overlay[roi.y1:roi.y2, roi.x1:roi.x2] = cv2.addWeighted(roi_img, 1.0, green, 0.5, 0.0)
    vis = overlay

    # bbox
    x1, y1, x2, y2 = result["bbox_xyxy"]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # centroid
    u, v = result["centroid_uv"]
    cv2.circle(vis, (int(round(u)), int(round(v))), 6, (0, 0, 255), -1)

    # obb
    box = obb["box"].copy()
    box[:, 0] += roi.x1
    box[:, 1] += roi.y1
    cv2.polylines(vis, [box], True, (255, 0, 255), 2)

    # grasp line
    gx1, gy1, gx2, gy2 = result["grasp_line_xyxy"]
    cv2.line(
        vis,
        (int(round(gx1)), int(round(gy1))),
        (int(round(gx2)), int(round(gy2))),
        (255, 0, 0),
        3,
    )

    txt1 = f"depth={result['depth_at_centroid_mm']} mm"
    txt2 = f"major={result['major_axis_angle_deg']:.1f}"
    txt3 = f"grasp={result['grasp_angle_deg']:.1f}"

    cv2.putText(vis, txt1, (roi.x1 + 10, roi.y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, txt2, (roi.x1 + 10, roi.y1 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, txt3, (roi.x1 + 10, roi.y1 + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return result, vis


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, CAMERA.color_width, CAMERA.color_height, rs.format.bgr8, CAMERA.fps)
    config.enable_stream(rs.stream.depth, CAMERA.depth_width, CAMERA.depth_height, rs.format.z16, CAMERA.fps)

    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()

    print("depth_scale:", depth_scale)
    print("s: save / q: quit")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_mm = (depth_raw.astype(np.float32) * depth_scale * 1000.0).astype(np.uint16)

            result, vis = analyze_scene(color, depth_mm, intrinsics)

            if result is not None:
                print(json.dumps(result, ensure_ascii=False))

            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_raw, alpha=0.03),
                cv2.COLORMAP_JET
            )

            cv2.imshow("color_grasp", vis)
            cv2.imshow("depth_vis", depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                cv2.imwrite("saved_color.png", color)
                cv2.imwrite("saved_depth_raw.png", depth_raw)
                cv2.imwrite("saved_depth_vis.png", depth_vis)
                cv2.imwrite("saved_debug.png", vis)
                print("saved files")
            elif key == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()