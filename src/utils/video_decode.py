"""视频前处理工具。

本模块负责把手机拍摄的视频帧恢复成协议解码器可接受的标准二维码图像：
1. 逐帧读取视频；
2. 检测三个大定位块和右下角小定位块；
3. 做粗矫正与二次精修；
4. 输出 1080x1080 的规范化二维码帧序列。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

try:
    # 包内导入，供编辑器/静态分析器正确解析。
    from ..config import BIG_FINDER_SIZE, GRID_SIZE, QUIET_WIDTH, SMALL_FINDER_SIZE
except ImportError:
    # 兼容 `python src/decode.py ...` 这类直接脚本运行方式。
    from config import BIG_FINDER_SIZE, GRID_SIZE, QUIET_WIDTH, SMALL_FINDER_SIZE


UPSCALE = 10
CANVAS_SIZE = GRID_SIZE * UPSCALE
BIG_FINDER_TOPLEFTS = np.array(
    [
        [QUIET_WIDTH * UPSCALE, QUIET_WIDTH * UPSCALE],
        [(GRID_SIZE - QUIET_WIDTH - BIG_FINDER_SIZE) * UPSCALE, QUIET_WIDTH * UPSCALE],
        [QUIET_WIDTH * UPSCALE, (GRID_SIZE - QUIET_WIDTH - BIG_FINDER_SIZE) * UPSCALE],
    ],
    dtype=np.float32,
)
SMALL_FINDER_TOPLEFT = np.array(
    [
        (GRID_SIZE - QUIET_WIDTH - SMALL_FINDER_SIZE) * UPSCALE,
        (GRID_SIZE - QUIET_WIDTH - SMALL_FINDER_SIZE) * UPSCALE,
    ],
    dtype=np.float32,
)
BIG_FINDER_CENTERS = BIG_FINDER_TOPLEFTS + (BIG_FINDER_SIZE * UPSCALE) / 2.0
EXPECTED_SMALL_CENTER = SMALL_FINDER_TOPLEFT + (SMALL_FINDER_SIZE * UPSCALE) / 2.0


@dataclass
class FinderCandidate:
    """定位块候选区域。

    center: 候选块中心点
    size: 候选块平均边长
    box: 最小外接旋转矩形的四个顶点
    """

    center: np.ndarray
    size: float
    box: np.ndarray


def extract_video_frames(path: str) -> list[np.ndarray]:
    """顺序读取视频中的所有帧。"""

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"failed to open video: {path}")

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)

    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from video: {path}")
    return frames


def video_to_qr_sequence(path: str) -> list[np.ndarray]:
    """将输入视频转换为可直接解码的二维码帧序列。"""

    frames = extract_video_frames(path)
    qr_frames: list[np.ndarray] = []
    for frame in frames:
        rectified = rectify_frame(frame)
        if rectified is not None:
            qr_frames.append(rectified)
    if not qr_frames:
        raise RuntimeError("no QR-like frames were rectified from the input video")
    return qr_frames


def rectify_frame(frame: np.ndarray) -> np.ndarray | None:
    """对单帧做定位与几何矫正。

    优先路径：
    - 直接在原图找到 3 个大定位块 + 1 个小定位块，做四点透视变换。

    回退路径：
    - 只用 3 个大定位块先做粗仿射矫正；
    - 再在规范坐标系中重新定位四个定位块并二次精修。
    """

    gray = _to_gray(frame)
    candidates = _find_finder_candidates(gray, min_side=12.0)
    ordered = _select_big_finders(candidates)
    if ordered is None:
        return None

    direct: np.ndarray | None = None
    small = _select_small_finder(candidates, ordered)
    if small is not None:
        direct = _warp_perspective(gray, ordered, small)
        refined = _refine_in_canonical_space(direct)
        if refined is not None:
            return refined

    coarse = _warp_coarse(gray, ordered)
    refined = _refine_in_canonical_space(coarse)
    if refined is not None:
        return refined
    return direct if direct is not None else coarse


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """统一转成灰度图，减少后续阈值和轮廓处理复杂度。"""

    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _find_finder_candidates(
    gray: np.ndarray,
    *,
    min_side: float,
    max_side: float | None = None,
) -> list[FinderCandidate]:
    """在图像中筛出“像定位块”的候选区域。"""

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []

    resolved_max_side = max_side
    if resolved_max_side is None:
        resolved_max_side = min(gray.shape[:2]) * 0.45

    raw: list[FinderCandidate] = []
    tree = hierarchy[0]
    for idx, contour in enumerate(contours):
        child = tree[idx][2]
        if child < 0 or tree[child][2] < 0:
            continue

        # 先用尺寸、长宽比、面积快速过滤，再用 1:1:3:1:1 纹理校验。
        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        if w <= 1 or h <= 1:
            continue

        side = max(w, h)
        short = min(w, h)
        if side < min_side or side > resolved_max_side:
            continue
        ratio = side / max(short, 1e-6)
        if ratio > 1.35:
            continue
        if cv2.contourArea(contour) < side * side * 0.15:
            continue
        if not _matches_finder_pattern(binary, rect):
            continue

        box = cv2.boxPoints(rect).astype(np.float32)
        raw.append(
            FinderCandidate(
                center=np.array(rect[0], dtype=np.float32),
                size=float((w + h) / 2.0),
                box=box,
            )
        )

    return _dedupe_candidates(raw)


def _matches_finder_pattern(binary: np.ndarray, rect) -> bool:
    """检查候选区域是否具有定位块的黑白比例特征。"""

    crop = _crop_rotated(binary, rect, 70)
    if crop is None:
        return False

    mids = []
    center = crop.shape[0] // 2
    mids.append(np.mean(crop[max(0, center - 2) : center + 3, :], axis=0))
    mids.append(np.mean(crop[:, max(0, center - 2) : center + 3], axis=1))
    return all(_check_ratio_line(line > 127) for line in mids)


def _crop_rotated(image: np.ndarray, rect, size: int) -> np.ndarray | None:
    """将旋转矩形拉正成固定大小，方便做比例模式检测。"""

    box = cv2.boxPoints(rect).astype(np.float32)
    w, h = rect[1]
    if w <= 1 or h <= 1:
        return None

    if w >= h:
        src = _order_box_points(box)
    else:
        src = np.roll(_order_box_points(box), -1, axis=0)

    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (size, size), borderValue=0)


def _order_box_points(box: np.ndarray) -> np.ndarray:
    """将矩形四点整理为左上、右上、右下、左下。"""

    pts = np.asarray(box, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = pts[:, 0] - pts[:, 1]
    ordered = np.empty((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmax(diffs)]
    ordered[3] = pts[np.argmin(diffs)]
    return ordered


def _check_ratio_line(line: np.ndarray) -> bool:
    """检查一条采样线是否接近定位块的 1:1:3:1:1 比例。"""

    values = line.astype(np.uint8).tolist()
    if not values:
        return False

    runs: list[tuple[int, int]] = []
    current = values[0]
    count = 1
    for value in values[1:]:
        if value == current:
            count += 1
        else:
            runs.append((current, count))
            current = value
            count = 1
    runs.append((current, count))

    if len(runs) < 5:
        return False

    best_score = float("inf")
    for start in range(len(runs) - 4):
        window = runs[start : start + 5]
        if window[2][1] != max(length for _, length in window):
            continue
        lengths = np.array([length for _, length in window], dtype=np.float32)
        base = lengths[2] / 3.0
        if base <= 0:
            continue
        ratios = lengths / base
        score = float(np.abs(ratios - np.array([1, 1, 3, 1, 1], dtype=np.float32)).sum())
        if score < best_score:
            best_score = score

    return best_score < 2.3


def _dedupe_candidates(candidates: list[FinderCandidate]) -> list[FinderCandidate]:
    """去掉中心点非常接近的重复候选块。"""

    unique: list[FinderCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.size, reverse=True):
        if any(np.linalg.norm(candidate.center - kept.center) < max(candidate.size, kept.size) * 0.35 for kept in unique):
            continue
        unique.append(candidate)
    return unique


def _select_big_finders(candidates: list[FinderCandidate]) -> tuple[FinderCandidate, FinderCandidate, FinderCandidate] | None:
    """从候选块中选出左上、右上、左下三个大定位块。"""

    if len(candidates) < 3:
        return None

    best_score = float("inf")
    best_triplet: tuple[FinderCandidate, FinderCandidate, FinderCandidate] | None = None

    for triplet in combinations(candidates, 3):
        sizes = [item.size for item in triplet]
        if min(sizes) <= 0:
            continue
        if max(sizes) / min(sizes) > 1.45:
            continue

        # 三个大定位块应形成接近直角的几何关系。
        centers = [item.center for item in triplet]
        angle_idx, angle_delta = _right_angle_index(centers)
        if angle_idx < 0 or angle_delta > 22.0:
            continue

        score = angle_delta * 10.0 + (max(sizes) - min(sizes))
        if score < best_score:
            ordered = _order_triplet_by_geometry(triplet, angle_idx)
            if ordered is not None:
                best_score = score
                best_triplet = ordered

    return best_triplet


def _right_angle_index(points: list[np.ndarray]) -> tuple[int, float]:
    """返回三个点中最接近直角的顶点下标及其偏差。"""

    best_idx = -1
    best_delta = float("inf")
    for idx in range(3):
        others = [points[i] for i in range(3) if i != idx]
        v1 = others[0] - points[idx]
        v2 = others[1] - points[idx]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom <= 1e-6:
            continue
        cosv = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cosv)))
        delta = abs(angle - 90.0)
        if delta < best_delta:
            best_idx = idx
            best_delta = delta
    return best_idx, best_delta


def _order_triplet_by_geometry(
    triplet: tuple[FinderCandidate, FinderCandidate, FinderCandidate],
    angle_idx: int,
) -> tuple[FinderCandidate, FinderCandidate, FinderCandidate] | None:
    """根据叉积方向统一三个大定位块的顺序。"""

    tl = triplet[angle_idx]
    others = [triplet[i] for i in range(3) if i != angle_idx]
    v1 = others[0].center - tl.center
    v2 = others[1].center - tl.center
    cross = float(v1[0] * v2[1] - v1[1] * v2[0])
    if abs(cross) < 1e-6:
        return None
    if cross > 0:
        return tl, others[0], others[1]
    return tl, others[1], others[0]


def _warp_coarse(
    gray: np.ndarray,
    ordered: tuple[FinderCandidate, FinderCandidate, FinderCandidate],
) -> np.ndarray:
    """仅使用三个大定位块做一次粗仿射矫正。"""

    tl, tr, bl = ordered
    x_axis = _normalize(tr.center - tl.center)
    y_axis = _normalize(bl.center - tl.center)

    source = np.array(
        [
            _finder_topleft_corner(tl, x_axis, y_axis),
            _finder_topleft_corner(tr, x_axis, y_axis),
            _finder_topleft_corner(bl, x_axis, y_axis),
        ],
        dtype=np.float32,
    )
    matrix = cv2.getAffineTransform(source, BIG_FINDER_TOPLEFTS)
    return cv2.warpAffine(gray, matrix, (CANVAS_SIZE, CANVAS_SIZE), flags=cv2.INTER_LINEAR, borderValue=255)


def _select_small_finder(
    candidates: list[FinderCandidate],
    ordered: tuple[FinderCandidate, FinderCandidate, FinderCandidate],
) -> FinderCandidate | None:
    """按位置和尺寸从候选中选择右下角小定位块。"""

    tl, tr, bl = ordered
    avg_big = (tl.size + tr.size + bl.size) / 3.0
    predicted = tr.center + bl.center - tl.center

    best_score = float("inf")
    best: FinderCandidate | None = None
    for candidate in candidates:
        if any(np.linalg.norm(candidate.center - finder.center) < avg_big * 0.35 for finder in ordered):
            continue
        if candidate.size >= avg_big * 0.8:
            continue

        distance_score = float(np.linalg.norm(candidate.center - predicted)) / max(avg_big, 1e-6)
        size_score = abs(candidate.size / max(avg_big, 1e-6) - 0.5) * 4.0
        score = distance_score + size_score
        if score < best_score:
            best_score = score
            best = candidate

    if best_score > 3.5:
        return None
    return best


def _warp_perspective(
    gray: np.ndarray,
    ordered: tuple[FinderCandidate, FinderCandidate, FinderCandidate],
    small: FinderCandidate,
) -> np.ndarray:
    """使用 3 大 + 1 小 四个定位块直接做透视矫正。"""

    tl, tr, bl = ordered
    x_axis = _normalize(tr.center - tl.center)
    y_axis = _normalize(bl.center - tl.center)

    source = np.array(
        [
            _finder_topleft_corner(tl, x_axis, y_axis),
            _finder_topleft_corner(tr, x_axis, y_axis),
            _finder_topleft_corner(bl, x_axis, y_axis),
            _finder_topleft_corner(small, x_axis, y_axis),
        ],
        dtype=np.float32,
    )
    destination = np.array(
        [
            BIG_FINDER_TOPLEFTS[0],
            BIG_FINDER_TOPLEFTS[1],
            BIG_FINDER_TOPLEFTS[2],
            SMALL_FINDER_TOPLEFT,
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(gray, matrix, (CANVAS_SIZE, CANVAS_SIZE), flags=cv2.INTER_LINEAR, borderValue=255)


def _normalize(vec: np.ndarray) -> np.ndarray:
    """将方向向量归一化，便于后续投影计算。"""

    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _finder_topleft_corner(candidate: FinderCandidate, x_axis: np.ndarray, y_axis: np.ndarray) -> np.ndarray:
    """在候选块四个顶点中选出最接近“左上角”的那个点。"""

    best = candidate.box[0]
    best_score = float("inf")
    for point in candidate.box:
        delta = point - candidate.center
        score = float(np.dot(delta, x_axis) + np.dot(delta, y_axis))
        if score < best_score:
            best_score = score
            best = point
    return best.astype(np.float32)


def _refine_in_canonical_space(image: np.ndarray) -> np.ndarray | None:
    """在已经大致拉正的图上再做一次四点精修。"""

    candidates = _find_finder_candidates(image, min_side=20.0, max_side=220.0)
    if len(candidates) < 4:
        return None

    tl = _pick_nearest_finder(candidates, BIG_FINDER_CENTERS[0], BIG_FINDER_SIZE * UPSCALE)
    tr = _pick_nearest_finder(candidates, BIG_FINDER_CENTERS[1], BIG_FINDER_SIZE * UPSCALE, {id(tl)} if tl else None)
    bl = _pick_nearest_finder(
        candidates,
        BIG_FINDER_CENTERS[2],
        BIG_FINDER_SIZE * UPSCALE,
        {id(item) for item in (tl, tr) if item is not None},
    )
    used = {id(item) for item in (tl, tr, bl) if item is not None}
    small = _pick_nearest_finder(candidates, EXPECTED_SMALL_CENTER, SMALL_FINDER_SIZE * UPSCALE, used)
    if tl is None or tr is None or bl is None or small is None:
        return None

    source = np.array(
        [
            _canonical_topleft_corner(tl),
            _canonical_topleft_corner(tr),
            _canonical_topleft_corner(bl),
            _canonical_topleft_corner(small),
        ],
        dtype=np.float32,
    )
    destination = np.array(
        [
            BIG_FINDER_TOPLEFTS[0],
            BIG_FINDER_TOPLEFTS[1],
            BIG_FINDER_TOPLEFTS[2],
            SMALL_FINDER_TOPLEFT,
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, matrix, (CANVAS_SIZE, CANVAS_SIZE), flags=cv2.INTER_LINEAR, borderValue=255)


def _pick_nearest_finder(
    candidates: list[FinderCandidate],
    expected_center: np.ndarray,
    expected_size: float,
    used_ids: set[int] | None = None,
) -> FinderCandidate | None:
    """在规范坐标系中，为某个期望位置挑选最合适的定位块。"""

    if used_ids is None:
        used_ids = set()

    best_score = float("inf")
    best: FinderCandidate | None = None
    for candidate in candidates:
        if id(candidate) in used_ids:
            continue
        distance_score = float(np.linalg.norm(candidate.center - expected_center)) / max(expected_size, 1e-6)
        size_score = abs(candidate.size - expected_size) / max(expected_size, 1e-6)
        score = distance_score + size_score
        if score < best_score:
            best_score = score
            best = candidate

    if best_score > 1.6:
        return None
    return best


def _canonical_topleft_corner(candidate: FinderCandidate) -> np.ndarray:
    """规范坐标系下直接按左上顶点取点。"""

    return candidate.box[np.argmin(candidate.box.sum(axis=1))].astype(np.float32)
