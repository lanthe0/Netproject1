import argparse
import sys
from pathlib import Path
from itertools import combinations

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from config import GRID_SIZE, QUIET_WIDTH, RECTIFY_MODEL_PATH
except ImportError:
    try:
        from ..config import GRID_SIZE, QUIET_WIDTH, RECTIFY_MODEL_PATH
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from config import GRID_SIZE, QUIET_WIDTH, RECTIFY_MODEL_PATH


MODEL_PATH = RECTIFY_MODEL_PATH


def box_size(box_xyxy) -> tuple[float, float]:
    """计算检测框的宽和高。

    输入:
    - box_xyxy: `(x1, y1, x2, y2)` 形式的检测框。

    输出:
    - `(width, height)` 二元组。

    原理/流程:
    - 直接用右下角减去左上角，得到轴对齐包围框尺寸。
    """

    x1, y1, x2, y2 = box_xyxy
    return max(0.0, float(x2) - float(x1)), max(0.0, float(y2) - float(y1))


def box_side(box_xyxy) -> float:
    """估计检测框对应 finder 的平均边长。

    输入:
    - box_xyxy: `(x1, y1, x2, y2)` 形式的检测框。

    输出:
    - 宽高均值，作为 finder 尺寸的近似值。

    原理/流程:
    - 分别计算宽和高。
    - 取两者平均值，减少轻微拉伸带来的波动。
    """

    width, height = box_size(box_xyxy)
    return (width + height) * 0.5


def center_distance(det_a, det_b) -> float:
    """计算两个检测框中心点之间的欧氏距离。

    输入:
    - det_a: 第一个检测框字典。
    - det_b: 第二个检测框字典。

    输出:
    - 两个中心点之间的距离。

    原理/流程:
    - 先取两个框中心点。
    - 再计算二维平面上的欧氏距离。
    """

    return float(np.linalg.norm(center_of(det_a["xyxy"]) - center_of(det_b["xyxy"])))


def _detection_iou(det_a, det_b) -> float:
    """计算两个检测框的 IoU。

    输入:
    - det_a: 第一个检测框字典。
    - det_b: 第二个检测框字典。

    输出:
    - `[0, 1]` 范围内的 IoU 数值。

    原理/流程:
    - 计算两个轴对齐矩形的交并比。
    - 用于去重和抑制明显重复框。
    """

    ax1, ay1, ax2, ay2 = det_a["xyxy"]
    bx1, by1, bx2, by2 = det_b["xyxy"]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(1e-6, det_a["area"])
    area_b = max(1e-6, det_b["area"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _make_virtual_detection(center: np.ndarray, side: float, score: float) -> dict[str, object]:
    """根据预测中心构造一个虚拟小 finder 检测框。

    输入:
    - center: 预测得到的小 finder 中心点坐标。
    - side: 估计的小 finder 边长。
    - score: 分配给该虚拟框的置信分数。

    输出:
    - 与 YOLO 检测结果格式兼容的检测框字典。

    原理/流程:
    - 以预测中心为核心构造一个正方形框。
    - 当真实小 finder 漏检时，用该虚拟框支撑第一阶段几何矫正。
    """

    half = max(2.0, float(side) * 0.5)
    cx, cy = float(center[0]), float(center[1])
    x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
    return {
        "xyxy": (x1, y1, x2, y2),
        "score": float(score),
        "area": float((x2 - x1) * (y2 - y1)),
        "virtual": True,
    }


def prune_finder_candidates(detections, image_shape, max_candidates: int = 8):
    """对检测框做一次几何筛选，尽量去掉明显无用框。

    输入:
    - detections: 原始检测框列表。
    - image_shape: 图像形状，用于估计面积阈值。
    - max_candidates: 最多保留多少个候选框。

    输出:
    - 更干净、排序后的候选检测框列表。

    原理/流程:
    - 先按长宽比、面积和贴边程度过滤明显异常框。
    - 再融合 YOLO 分数、方形程度和面积先验做排序。
    - 最后通过 IoU 和中心距离做轻量去重，只保留少量高质量候选。
    """

    if not detections:
        return []

    image_h, image_w = image_shape[:2]
    image_area = float(image_h * image_w)
    sorted_areas = sorted((float(det["area"]) for det in detections), reverse=True)
    area_window = sorted_areas[: min(4, len(sorted_areas))]
    reference_area = float(np.median(area_window)) if area_window else 0.0
    min_area = max(image_area * 0.00015, reference_area * 0.12)
    max_area = min(image_area * 0.5, max(reference_area * 6.0, image_area * 0.03))
    pruned = []

    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        width, height = box_size(det["xyxy"])
        if width <= 0 or height <= 0:
            continue
        if det["area"] < min_area:
            continue
        if det["area"] > max_area:
            continue

        aspect = width / max(height, 1e-6)
        if not 0.55 <= aspect <= 1.45:
            continue

        edge_margin = min(x1, y1, image_w - x2, image_h - y2)
        square_score = 1.0 - min(abs(1.0 - aspect), 1.0)
        area_score = min(det["area"] / max(reference_area, 1e-6), 1.0)
        edge_score = 0.35 if edge_margin < min(image_w, image_h) * 0.015 else 0.0
        rank_score = float(det["score"]) * 2.0 + square_score + area_score - edge_score

        enriched = dict(det)
        enriched["rank_score"] = float(rank_score)
        pruned.append(enriched)

    pruned.sort(key=lambda item: (item["rank_score"], item["score"], item["area"]), reverse=True)

    deduped = []
    for det in pruned:
        keep = True
        for kept in deduped:
            if _detection_iou(det, kept) > 0.35:
                keep = False
                break
            limit = max(box_side(det["xyxy"]), box_side(kept["xyxy"])) * 0.35
            if center_distance(det, kept) < limit:
                keep = False
                break
        if keep:
            deduped.append(det)
        if len(deduped) >= max_candidates:
            break

    return deduped


def select_roles_with_prediction(detections, min_area_ratio: float = 1.2):
    """优先用三个大 finder 推算右下小 finder，并给出角色分配结果。

    输入:
    - detections: 已经过滤后的检测框列表。
    - min_area_ratio: 大 finder 与小 finder 的最小面积比先验。

    输出:
    - `roles` 字典，包含 `tl/tr/br/bl` 四个角色。

    原理/流程:
    - 枚举若干组三个候选框，假设它们对应三个大 finder。
    - 依据三角拓扑推算右下小 finder 的理论中心位置。
    - 优先从预测点附近挑真实小框；若附近没有可靠候选，则构造虚拟小框补位。
    """

    if len(detections) < 3:
        raise RuntimeError("Pass1 failed: less than 3 detections.")

    ordered = sorted(
        detections,
        key=lambda item: (float(item.get("rank_score", item["score"])), item["area"]),
        reverse=True,
    )
    search_pool = ordered[: min(len(ordered), 8)]
    image_score = None
    best_roles = None

    for combo in combinations(search_pool, 3):
        big3 = list(combo)
        big_pts = [center_of(det["xyxy"]) for det in big3]
        sums = [p[0] + p[1] for p in big_pts]
        tl_idx = int(np.argmin(sums))
        tl_det = big3[tl_idx]
        tl_center = big_pts[tl_idx]

        rem = [(big3[i], big_pts[i]) for i in range(3) if i != tl_idx]
        if rem[0][1][0] >= rem[1][1][0]:
            tr_det, tr_center = rem[0]
            bl_det, bl_center = rem[1]
        else:
            tr_det, tr_center = rem[1]
            bl_det, bl_center = rem[0]

        vec_tr = tr_center - tl_center
        vec_bl = bl_center - tl_center
        len_tr = float(np.linalg.norm(vec_tr))
        len_bl = float(np.linalg.norm(vec_bl))
        if len_tr < 8 or len_bl < 8:
            continue

        cosine = abs(float(np.dot(vec_tr, vec_bl)) / max(len_tr * len_bl, 1e-6))
        if cosine > 0.6:
            continue

        predicted_br = tr_center + bl_center - tl_center
        big_areas = [float(det["area"]) for det in (tl_det, tr_det, bl_det)]
        area_similarity = min(big_areas) / max(max(big_areas), 1e-6)
        avg_big_side = float(np.mean([box_side(det["xyxy"]) for det in (tl_det, tr_det, bl_det)]))
        expected_small_side = max(6.0, avg_big_side * 0.5)
        max_search_dist = max(avg_big_side * 1.6, 24.0)

        small_candidates = [det for det in search_pool if det not in big3]
        best_small = None
        best_small_score = -1e9
        for small_det in small_candidates:
            small_center = center_of(small_det["xyxy"])
            dist = float(np.linalg.norm(small_center - predicted_br))
            if dist > max_search_dist:
                continue

            small_area = float(small_det["area"])
            area_ratio = min(big_areas) / max(small_area, 1e-6)
            if area_ratio < min_area_ratio * 0.7:
                continue

            area_fit = 1.0 / (1.0 + abs(area_ratio - 2.5))
            dist_fit = 1.0 - min(dist / max_search_dist, 1.0)
            candidate_score = float(small_det["score"]) + area_fit + dist_fit
            if candidate_score > best_small_score:
                best_small_score = candidate_score
                best_small = small_det

        geometry_score = (
            float(tl_det["score"] + tr_det["score"] + bl_det["score"])
            + area_similarity * 2.0
            + (1.0 - cosine) * 2.0
        )

        if best_small is None:
            br_det = _make_virtual_detection(predicted_br, expected_small_side, score=0.01)
            total_score = geometry_score - 0.35
        else:
            br_det = best_small
            total_score = geometry_score + best_small_score

        roles = {
            "tl": tl_det,
            "tr": tr_det,
            "br": br_det,
            "bl": bl_det,
        }
        if image_score is None or total_score > image_score:
            image_score = total_score
            best_roles = roles

    if best_roles is None:
        raise RuntimeError("Pass1 failed: unable to build role assignment from detections.")
    return best_roles


def _contour_depth(hierarchy: np.ndarray, index: int) -> int:
    """计算某个轮廓被多少层父轮廓包围。

    输入:
    - hierarchy: OpenCV 返回的轮廓层级数组，形状为 `(n, 4)`。
    - index: 当前要检查的轮廓下标。

    输出:
    - 轮廓嵌套深度整数值。深度越大，越像 finder 那种黑白嵌套结构。

    原理/流程:
    - 沿着当前轮廓的 parent 指针不断向上回溯。
    - 统计一共经过了多少层父轮廓。
    """

    depth = 0
    parent = int(hierarchy[index][3])
    while parent >= 0:
        depth += 1
        parent = int(hierarchy[parent][3])
    return depth


def _preprocess_for_finder_detection(image: np.ndarray) -> np.ndarray:
    """构造有利于 finder 轮廓提取的二值图。

    输入:
    - image: 手机拍摄得到的 BGR 图像。

    输出:
    - 以深色结构为前景的二值图，便于后续做轮廓搜索。

    原理/流程:
    - 先转灰度并轻度模糊，压制噪声。
    - 再做自适应阈值，尽量保住不均匀光照下的黑白环结构。
    - 最后做一次闭运算，把断开的方形边缘重新连起来。
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5,
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)


def detect_finders_opencv(image: np.ndarray, max_candidates: int = 20):
    """使用传统视觉轮廓几何规则提取 finder 候选框。

    输入:
    - image: 包含投影二维码的 BGR 图像。
    - max_candidates: 最多保留多少个候选框。

    输出:
    - 与 YOLO 路径兼容的检测结果列表，元素格式为:
      `{"xyxy": (x1, y1, x2, y2), "score": float, "area": float}`。

    原理/流程:
    - 先构造突出黑白嵌套结构的二值图。
    - 再基于轮廓层级、近似方形、面积和填充率筛选候选。
    - 最后按嵌套深度和形状质量打分，并做简单去重。
    """

    binary = _preprocess_for_finder_detection(image)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or not contours:
        return []

    hierarchy = hierarchy[0]
    image_area = float(image.shape[0] * image.shape[1])
    candidates = []

    for index, contour in enumerate(contours):
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.0005:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(contour, 0.08 * perimeter, True)
        if len(approx) < 4 or len(approx) > 8:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue

        aspect = w / float(h)
        if not 0.65 <= aspect <= 1.35:
            continue

        rect_area = float(w * h)
        fill_ratio = area / max(rect_area, 1.0)
        if not 0.25 <= fill_ratio <= 0.95:
            continue

        depth = _contour_depth(hierarchy, index)
        child = int(hierarchy[index][2])
        has_child = child >= 0
        if depth < 1 and not has_child:
            continue

        score = depth * 2.0
        score += 1.0 if has_child else 0.0
        score += max(0.0, 1.0 - abs(1.0 - aspect))
        score += fill_ratio

        candidates.append(
            {
                "xyxy": (float(x), float(y), float(x + w), float(y + h)),
                "score": float(score),
                "area": rect_area,
            }
        )

    candidates.sort(key=lambda item: (item["score"], item["area"]), reverse=True)

    deduped = []
    for candidate in candidates:
        x1, y1, x2, y2 = candidate["xyxy"]
        keep = True
        for kept in deduped:
            kx1, ky1, kx2, ky2 = kept["xyxy"]
            inter_x1 = max(x1, kx1)
            inter_y1 = max(y1, ky1)
            inter_x2 = min(x2, kx2)
            inter_y2 = min(y2, ky2)
            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter = inter_w * inter_h
            union = candidate["area"] + kept["area"] - inter
            iou = inter / union if union > 0 else 0.0
            if iou > 0.5:
                keep = False
                break
        if keep:
            deduped.append(candidate)
        if len(deduped) >= max_candidates:
            break

    return prune_finder_candidates(deduped, image.shape, max_candidates=max_candidates)


def resize_for_display(image: np.ndarray, max_side: int = 1000) -> np.ndarray:
    h, w = image.shape[:2]
    cur_max = max(h, w)
    if cur_max <= max_side:
        return image

    scale = max_side / float(cur_max)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


def detect_finders(model: YOLO, image: np.ndarray, conf: float, iou: float, max_det: int):
    results = model(image, conf=conf, iou=iou, max_det=max_det, verbose=False)
    if not results:
        return []

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    scores = boxes.conf.detach().cpu().numpy()

    out = []
    for box, score in zip(xyxy, scores):
        x1, y1, x2, y2 = box.tolist()
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        out.append(
            {
                "xyxy": (float(x1), float(y1), float(x2), float(y2)),
                "score": float(score),
                "area": float(area),
            }
        )
    return prune_finder_candidates(out, image.shape, max_candidates=max_det)


def center_of(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


def pick_point_by_role(points: np.ndarray, role: str) -> np.ndarray:
    if role == "tl":
        idx = int(np.argmin(points[:, 0] + points[:, 1]))
    elif role == "tr":
        idx = int(np.argmax(points[:, 0] - points[:, 1]))
    elif role == "br":
        idx = int(np.argmax(points[:, 0] + points[:, 1]))
    elif role == "bl":
        idx = int(np.argmax(points[:, 1] - points[:, 0]))
    else:
        raise ValueError(f"Unknown role: {role}")
    return points[idx].astype(np.float32)


def bbox_corner_for_role(box_xyxy, role: str) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy
    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    return pick_point_by_role(pts, role)


def refine_corner_from_box(image: np.ndarray, box_xyxy, role: str, expand_ratio: float = 0.18) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    ex = bw * max(0.0, expand_ratio)
    ey = bh * max(0.0, expand_ratio)

    rx1 = int(max(0, np.floor(x1 - ex)))
    ry1 = int(max(0, np.floor(y1 - ey)))
    rx2 = int(min(w - 1, np.ceil(x2 + ex)))
    ry2 = int(min(h - 1, np.ceil(y2 + ey)))

    if rx2 <= rx1 or ry2 <= ry1:
        return bbox_corner_for_role(box_xyxy, role)

    roi = image[ry1 : ry2 + 1, rx1 : rx2 + 1]
    if roi.size == 0:
        return bbox_corner_for_role(box_xyxy, role)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(bw_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bbox_corner_for_role(box_xyxy, role)

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 16:
        return bbox_corner_for_role(box_xyxy, role)

    rect = cv2.minAreaRect(cnt)
    pts = cv2.boxPoints(rect).astype(np.float32)
    pts[:, 0] += rx1
    pts[:, 1] += ry1
    return pick_point_by_role(pts, role)


def build_quad_from_roles(image: np.ndarray, roles, refine: bool, expand_ratio: float) -> np.ndarray:
    order = ["tl", "tr", "br", "bl"]
    pts = []
    for role in order:
        box = roles[role]["xyxy"]
        if refine:
            p = refine_corner_from_box(image, box, role, expand_ratio=expand_ratio)
        else:
            p = bbox_corner_for_role(box, role)
        pts.append(p)
    return np.array(pts, dtype=np.float32)


def pick_3_big_1_small(detections, min_area_ratio=1.2):
    if len(detections) < 4:
        return None

    top4 = sorted(detections, key=lambda d: d["area"], reverse=True)[:4]
    top4_sorted = sorted(top4, key=lambda d: d["area"], reverse=True)

    big3 = top4_sorted[:3]
    small1 = top4_sorted[3]

    area_ratio = big3[2]["area"] / max(small1["area"], 1e-6)
    if area_ratio < min_area_ratio:
        return None

    return big3, small1


def assign_roles(big3, small1):
    big_pts = [center_of(d["xyxy"]) for d in big3]

    # TL: smallest x+y
    sums = [p[0] + p[1] for p in big_pts]
    tl_idx = int(np.argmin(sums))
    tl_det = big3[tl_idx]
    tl_center = big_pts[tl_idx]

    rem = [(big3[i], big_pts[i]) for i in range(3) if i != tl_idx]
    if rem[0][1][0] >= rem[1][1][0]:
        tr_det = rem[0][0]
        bl_det = rem[1][0]
        tr_center = rem[0][1]
        bl_center = rem[1][1]
    else:
        tr_det = rem[1][0]
        bl_det = rem[0][0]
        tr_center = rem[1][1]
        bl_center = rem[0][1]

    br_det = small1
    br_center = center_of(small1["xyxy"])

    centers = np.array([tl_center, tr_center, br_center, bl_center], dtype=np.float32)
    roles = {
        "tl": tl_det,
        "tr": tr_det,
        "br": br_det,
        "bl": bl_det,
    }
    return centers, roles


def assign_roles_from_four(dets4):
    # Fallback when size-based 3-big-1-small split is unstable.
    pts = [center_of(d["xyxy"]) for d in dets4]

    tl_idx = int(np.argmin([p[0] + p[1] for p in pts]))
    br_idx = int(np.argmax([p[0] + p[1] for p in pts]))

    rem = [i for i in range(4) if i not in (tl_idx, br_idx)]
    i1, i2 = rem[0], rem[1]
    # x-y larger tends to be TR, the other is BL.
    if (pts[i1][0] - pts[i1][1]) >= (pts[i2][0] - pts[i2][1]):
        tr_idx, bl_idx = i1, i2
    else:
        tr_idx, bl_idx = i2, i1

    centers = np.array([pts[tl_idx], pts[tr_idx], pts[br_idx], pts[bl_idx]], dtype=np.float32)
    roles = {
        "tl": dets4[tl_idx],
        "tr": dets4[tr_idx],
        "br": dets4[br_idx],
        "bl": dets4[bl_idx],
    }
    return centers, roles


def build_dst_points(out_size: int, center_margin_ratio: float) -> np.ndarray:
    margin = int(round(out_size * max(0.0, min(0.45, center_margin_ratio))))
    return np.array(
        [
            [margin, margin],
            [out_size - 1 - margin, margin],
            [out_size - 1 - margin, out_size - 1 - margin],
            [margin, out_size - 1 - margin],
        ],
        dtype=np.float32,
    )


def build_decoder_dst_points(out_size: int, expand_modules: float = 0.0) -> np.ndarray:
    """构造解码专用透视变换的目标四角坐标。

    输入:
    - out_size: 矫正输出图像边长，通常为 `GRID_SIZE * module_pixels`。
    - expand_modules: 在协议边界基础上，额外向外扩多少个模块宽度。

    输出:
    - `4 x 2` 的浮点坐标，表示 TL/TR/BR/BL 四个目标点。

    原理/流程:
    - 基础位置仍对齐到协议里 quiet zone 内侧边界。
    - 可选地向外扩一小段模块宽度，用于生成多个 decoder 候选几何。
    - 不同扩张量可以覆盖不同帧上的系统性内缩偏差。
    """

    scale = out_size / float(GRID_SIZE)
    expand = expand_modules * scale
    inset = QUIET_WIDTH * scale - expand
    far = (GRID_SIZE - QUIET_WIDTH) * scale + expand
    return np.array(
        [
            [inset, inset],
            [far, inset],
            [far, far],
            [inset, far],
        ],
        dtype=np.float32,
    )


def _build_center_alignment_pattern() -> np.ndarray:
    """构造协议中心小定位格的 5x5 二值模板。

    输入:
    - 无，模板由协议固定定义。

    输出:
    - `5x5` 的 `uint8` 二值模板，`1` 表示黑色模块。

    原理/流程:
    - 外边框为黑色；
    - 内层保留白色；
    - 中心单点为黑色；
    - 与 `_2Dcode.make_base_grid()` 中的中心小定位格保持一致。
    """

    pattern = np.zeros((5, 5), dtype=np.uint8)
    pattern[0, :] = 1
    pattern[-1, :] = 1
    pattern[:, 0] = 1
    pattern[:, -1] = 1
    pattern[2, 2] = 1
    return pattern


CENTER_ALIGNMENT_PATTERN = _build_center_alignment_pattern()
CENTER_ALIGNMENT_SIZE = int(CENTER_ALIGNMENT_PATTERN.shape[0])
CENTER_ALIGNMENT_START = (GRID_SIZE - CENTER_ALIGNMENT_SIZE) // 2
CENTER_ALIGNMENT_CENTER = CENTER_ALIGNMENT_START + CENTER_ALIGNMENT_SIZE * 0.5


def detect_center_alignment_anchor(
    rectified: np.ndarray,
    *,
    search_modules: float = 2.0,
    sample_ratio: float = 0.55,
) -> dict[str, object]:
    """在 decoder 矫正图中检测中心小定位格，并估计其偏移量。

    输入:
    - rectified: 已映射到 decoder 协议坐标的矫正图。
    - search_modules: 在理论中心附近，最多搜索多少个模块宽度。
    - sample_ratio: 每个模块内部用于判断黑白的中心采样比例。

    输出:
    - 字典，包含：
      - `found`: 是否找到可靠的小定位格
      - `score`: 最优模板匹配分数，范围约为 `[0, 1]`
      - `offset_px`: 估计出的 `(dx, dy)` 像素偏移
      - `center_px`: 估计出的中心点像素坐标
      - `module`: 当前图像推断出的模块像素边长

    原理/流程:
    - 协议中间存在固定的 `5x5` 小定位格，可作为内部锚点；
    - 在理论中心附近做小范围平移搜索；
    - 对每个候选位置按模块采样，和固定模板比较黑白误差；
    - 选出误差最小的位置，并输出相对理论中心的偏移。
    """

    if rectified is None or rectified.size == 0:
        return {
            "found": False,
            "score": 0.0,
            "offset_px": np.zeros(2, dtype=np.float32),
            "center_px": np.zeros(2, dtype=np.float32),
            "module": 0,
        }

    if rectified.ndim == 2:
        gray = rectified
    else:
        gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape[:2]
    module = max(1, min(height, width) // GRID_SIZE)
    expected_left = CENTER_ALIGNMENT_START * module
    expected_top = CENTER_ALIGNMENT_START * module
    search_px = max(1, int(round(search_modules * module)))
    margin = max(1, int(round(module * (1.0 - sample_ratio) * 0.5)))

    best_score = -1.0
    best_offset = np.zeros(2, dtype=np.float32)

    for dy in range(-search_px, search_px + 1):
        for dx in range(-search_px, search_px + 1):
            mismatch = 0.0
            weight_sum = 0.0
            valid = True
            for row in range(CENTER_ALIGNMENT_SIZE):
                for col in range(CENTER_ALIGNMENT_SIZE):
                    top = expected_top + dy + row * module + margin
                    left = expected_left + dx + col * module + margin
                    bottom = expected_top + dy + (row + 1) * module - margin
                    right = expected_left + dx + (col + 1) * module - margin

                    if top < 0 or left < 0 or bottom > height or right > width or top >= bottom or left >= right:
                        valid = False
                        break

                    patch = gray[top:bottom, left:right]
                    norm_gray = float(patch.mean()) / 255.0
                    expected_black = float(CENTER_ALIGNMENT_PATTERN[row, col])
                    expected_gray = 0.0 if expected_black > 0.5 else 1.0
                    weight = 1.5 if (row == 2 and col == 2) else 1.0
                    mismatch += abs(norm_gray - expected_gray) * weight
                    weight_sum += weight
                if not valid:
                    break

            if not valid or weight_sum <= 0.0:
                continue

            score = 1.0 - mismatch / weight_sum
            if score > best_score:
                best_score = score
                best_offset = np.array([dx, dy], dtype=np.float32)

    expected_center = np.array(
        [CENTER_ALIGNMENT_CENTER * module, CENTER_ALIGNMENT_CENTER * module],
        dtype=np.float32,
    )
    center_px = expected_center + best_offset
    return {
        "found": bool(best_score >= 0.72),
        "score": float(max(best_score, 0.0)),
        "offset_px": best_offset,
        "center_px": center_px,
        "module": int(module),
    }


def refine_decoder_with_center_anchor(
    rectified: np.ndarray,
    *,
    score_threshold: float = 0.78,
    max_shift_modules: float = 1.6,
) -> np.ndarray:
    """利用中心小定位格对 decoder 矫正图做一次轻量平移微调。

    输入:
    - rectified: 初步完成 decoder 映射的矫正图。
    - score_threshold: 只有中心锚点检测分数达到该阈值时才执行微调。
    - max_shift_modules: 允许的最大平移量，单位为模块宽度。

    输出:
    - 若检测稳定，则返回平移微调后的图像；否则原样返回。

    原理/流程:
    - 中间小定位格能反映内部网格是否与 payload 中心区域对齐；
    - 先估计锚点相对理论中心的偏移；
    - 若偏移量不大且检测分数足够高，则只做一次整图平移修正；
    - 先用这种低风险方式接入内部锚点，避免直接改动主几何模型。
    """

    anchor = detect_center_alignment_anchor(rectified)
    if not anchor["found"] or float(anchor["score"]) < score_threshold:
        return rectified

    offset = np.asarray(anchor["offset_px"], dtype=np.float32)
    module = max(1.0, float(anchor["module"]))
    if float(np.linalg.norm(offset)) > max_shift_modules * module:
        return rectified

    translate = np.array(
        [
            [1.0, 0.0, -float(offset[0])],
            [0.0, 1.0, -float(offset[1])],
        ],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        rectified,
        translate,
        (rectified.shape[1], rectified.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def rectify_image(image: np.ndarray, src_pts: np.ndarray, out_size: int, center_margin_ratio: float):
    # Map center points to an inset quad (not image corners) to keep more outer context.
    dst_pts = build_dst_points(out_size, center_margin_ratio)
    mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    rectified = cv2.warpPerspective(image, mat, (out_size, out_size))
    return rectified, mat


def rectify_image_cropped(image: np.ndarray, src_pts: np.ndarray, out_size: int) -> np.ndarray:
    # Direct crop-style rectification from four refined corners.
    dst_pts = np.array(
        [[0, 0], [out_size - 1, 0], [out_size - 1, out_size - 1], [0, out_size - 1]],
        dtype=np.float32,
    )
    mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, mat, (out_size, out_size))


def rectify_image_for_decoder(
    image: np.ndarray,
    src_pts: np.ndarray,
    out_size: int,
    *,
    expand_modules: float = 0.0,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """按协议几何将四角映射到 decoder 采样坐标。

    输入:
    - image: 原始或阶段一矫正后的 BGR 图像。
    - src_pts: TL/TR/BR/BL 顺序的四个源角点。
    - out_size: 输出正方形图像边长。
    - expand_modules: 目标协议边界向外扩多少个模块，用于生成几何候选。
    - interpolation: 透视变换插值方式。黑白链路更能容忍线性插值，彩色链路更适合最近邻。

    输出:
    - 已映射到协议 quiet zone 几何上的 decoder 图像。

    原理/流程:
    - 先根据协议 quiet zone 计算目标四角位置；
    - 再用透视变换把 finder 角点映射到这些目标点；
    - 插值方式可配置，便于在黑白与彩色协议之间做不同优化。
    """

    dst_pts = build_decoder_dst_points(out_size, expand_modules=expand_modules)
    mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(
        image,
        mat,
        (out_size, out_size),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def warp_keep_all(image: np.ndarray, homography: np.ndarray, out_size: int) -> np.ndarray:
    # Apply homography while fitting the whole transformed image into output canvas.
    h, w = image.shape[:2]
    corners = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )
    tc = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), homography).reshape(-1, 2)

    min_x = float(np.min(tc[:, 0]))
    min_y = float(np.min(tc[:, 1]))
    max_x = float(np.max(tc[:, 0]))
    max_y = float(np.max(tc[:, 1]))

    span_w = max(max_x - min_x, 1e-6)
    span_h = max(max_y - min_y, 1e-6)
    scale = min((out_size - 1) / span_w, (out_size - 1) / span_h)

    tx = -min_x * scale + ((out_size - 1) - span_w * scale) * 0.5
    ty = -min_y * scale + ((out_size - 1) - span_h * scale) * 0.5

    fit = np.array(
        [
            [scale, 0.0, tx],
            [0.0, scale, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    composed = fit @ homography
    return cv2.warpPerspective(image, composed, (out_size, out_size), borderMode=cv2.BORDER_REPLICATE)


def draw_debug(image: np.ndarray, detections, src_pts: np.ndarray):
    vis = image.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            vis,
            f"{d['score']:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    colors = [(0, 255, 0), (0, 255, 255), (0, 128, 255), (255, 255, 0)]
    names = ["TL(center)", "TR(center)", "BR(center)", "BL(center)"]
    for i, p in enumerate(src_pts):
        x, y = int(p[0]), int(p[1])
        cv2.circle(vis, (x, y), 6, colors[i], -1)
        cv2.putText(
            vis,
            names[i],
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colors[i],
            2,
            cv2.LINE_AA,
        )
    return vis


def draw_role_points(image: np.ndarray, role_pts: np.ndarray, label: str):
    vis = image.copy()
    colors = [(0, 255, 0), (0, 255, 255), (0, 128, 255), (255, 255, 0)]
    names = ["TL", "TR", "BR", "BL"]
    for i, p in enumerate(role_pts):
        x, y = int(p[0]), int(p[1])
        cv2.circle(vis, (x, y), 6, colors[i], -1)
        cv2.putText(
            vis,
            names[i],
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colors[i],
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        vis,
        label,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return vis


def draw_detection_only(image: np.ndarray, detections, title: str = "pass2"):
    vis = image.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(
            vis,
            f"{d['score']:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        vis,
        f"{title}: {len(detections)} boxes",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 180, 255),
        2,
        cv2.LINE_AA,
    )
    return vis


def main():
    parser = argparse.ArgumentParser(description="Detect 3 big + 1 small finder and rectify perspective.")
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save all four output images")
    parser.add_argument("--size", type=int, default=1024, help="Rectified output size")
    parser.add_argument(
        "--center-margin-ratio",
        type=float,
        default=0.18,
        help="Margin ratio for mapping pass1 center points in rectified image (higher keeps more outer area)",
    )
    parser.add_argument(
        "--stage2-margin-ratio",
        type=float,
        default=0.0,
        help="Target margin ratio for second rectification. Uses keep-all warp to avoid extra crop.",
    )
    parser.add_argument(
        "--refine-corners",
        action="store_true",
        default=True,
        help="Refine QR corner points from finder ROI geometry (default on)",
    )
    parser.add_argument(
        "--corner-expand-ratio",
        type=float,
        default=0.18,
        help="ROI expand ratio when refining corners inside detection boxes",
    )
    parser.add_argument("--conf", type=float, default=0.1, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=20, help="Max detections")
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=1.2,
        help="Minimum ratio between smallest big finder area and small finder area",
    )
    parser.add_argument("--show", action="store_true", help="Display debug and rectified windows")
    parser.add_argument(
        "--show-max-side",
        type=int,
        default=1000,
        help="Max side length for display windows only",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Put best.pt in current directory or update MODEL_PATH in rectify.py"
        )

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)
    detections = detect_finders(model, image, conf=args.conf, iou=args.iou, max_det=args.max_det)

    picked = pick_3_big_1_small(detections, min_area_ratio=args.min_area_ratio)
    used_fallback = False
    if picked is None:
        if len(detections) < 4:
            raise RuntimeError("Failed to find enough finder detections for rectification.")
        top4 = sorted(detections, key=lambda d: d["score"], reverse=True)[:4]
        center_pts, roles = assign_roles_from_four(top4)
        used_fallback = True
    else:
        big3, small1 = picked
        center_pts, roles = assign_roles(big3, small1)
    stage1_corner_pts = build_quad_from_roles(
        image,
        roles,
        refine=args.refine_corners,
        expand_ratio=args.corner_expand_ratio,
    )

    rectified_stage1, _ = rectify_image(
        image,
        stage1_corner_pts,
        out_size=args.size,
        center_margin_ratio=args.center_margin_ratio,
    )
    debug_vis = draw_debug(image, detections, center_pts)
    debug_vis = draw_role_points(debug_vis, stage1_corner_pts, "stage1 corner points")

    pass2_detections = detect_finders(model, rectified_stage1, conf=args.conf, iou=args.iou, max_det=args.max_det)
    pass2_vis = draw_detection_only(rectified_stage1, pass2_detections, title="pass2")

    # Use second-pass centers for one more homography to reduce residual skew.
    rectified_stage2 = rectified_stage1
    pass2_refined = False
    pass2_picked = pick_3_big_1_small(pass2_detections, min_area_ratio=args.min_area_ratio)
    pass2_corner_pts = None
    if pass2_picked is not None:
        pass2_big3, pass2_small1 = pass2_picked
        _, pass2_roles = assign_roles(pass2_big3, pass2_small1)
        pass2_corner_pts = build_quad_from_roles(
            rectified_stage1,
            pass2_roles,
            refine=args.refine_corners,
            expand_ratio=args.corner_expand_ratio,
        )
        stage2_dst = build_dst_points(args.size, args.stage2_margin_ratio)
        stage2_h = cv2.getPerspectiveTransform(pass2_corner_pts, stage2_dst)
        rectified_stage2 = warp_keep_all(rectified_stage1, stage2_h, out_size=args.size)
        pass2_refined = True
    elif len(pass2_detections) >= 4:
        pass2_top4 = sorted(pass2_detections, key=lambda d: d["score"], reverse=True)[:4]
        _, pass2_roles = assign_roles_from_four(pass2_top4)
        pass2_corner_pts = build_quad_from_roles(
            rectified_stage1,
            pass2_roles,
            refine=args.refine_corners,
            expand_ratio=args.corner_expand_ratio,
        )
        stage2_dst = build_dst_points(args.size, args.stage2_margin_ratio)
        stage2_h = cv2.getPerspectiveTransform(pass2_corner_pts, stage2_dst)
        rectified_stage2 = warp_keep_all(rectified_stage1, stage2_h, out_size=args.size)
        pass2_refined = True

    # Extra outputs for cropped viewing and decoder-oriented geometry.
    if pass2_corner_pts is not None:
        refined_corner_cropped = rectify_image_cropped(rectified_stage1, pass2_corner_pts, out_size=args.size)
        decoder_rectified = rectify_image_for_decoder(rectified_stage1, pass2_corner_pts, out_size=args.size)
    else:
        refined_corner_cropped = rectify_image_cropped(image, stage1_corner_pts, out_size=args.size)
        decoder_rectified = rectify_image_for_decoder(image, stage1_corner_pts, out_size=args.size)

    first_detect_path = output_dir / "01_first_detection.jpg"
    first_rectified_path = output_dir / "02_first_rectified.jpg"
    second_detect_path = output_dir / "03_second_detection.jpg"
    second_rectified_path = output_dir / "04_second_rectified.jpg"
    refined_corner_cropped_path = output_dir / "05_refined_corner_cropped.jpg"
    decoder_rectified_path = output_dir / "06_decoder_rectified.jpg"

    cv2.imwrite(str(first_detect_path), debug_vis)
    cv2.imwrite(str(first_rectified_path), rectified_stage1)
    cv2.imwrite(str(second_detect_path), pass2_vis)
    cv2.imwrite(str(second_rectified_path), rectified_stage2)
    cv2.imwrite(str(refined_corner_cropped_path), refined_corner_cropped)
    cv2.imwrite(str(decoder_rectified_path), decoder_rectified)

    print(f"Model: {MODEL_PATH}")
    print(f"Input: {image_path.resolve()}")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Pass1 boxes: {len(detections)}")
    print(f"Pass1 fallback-4pt used: {used_fallback}")
    print(f"Pass2 boxes: {len(pass2_detections)}")
    print(f"Pass2 refine applied: {pass2_refined}")
    print(f"Saved: {first_detect_path.resolve()}")
    print(f"Saved: {first_rectified_path.resolve()}")
    print(f"Saved: {second_detect_path.resolve()}")
    print(f"Saved: {second_rectified_path.resolve()}")
    print(f"Saved: {refined_corner_cropped_path.resolve()}")
    print(f"Saved: {decoder_rectified_path.resolve()}")

    if args.show:
        debug_show = resize_for_display(debug_vis, max_side=args.show_max_side)
        rectified1_show = resize_for_display(rectified_stage1, max_side=args.show_max_side)
        pass2_show = resize_for_display(pass2_vis, max_side=args.show_max_side)
        rectified2_show = resize_for_display(rectified_stage2, max_side=args.show_max_side)
        refined_cropped_show = resize_for_display(refined_corner_cropped, max_side=args.show_max_side)
        decoder_show = resize_for_display(decoder_rectified, max_side=args.show_max_side)
        cv2.imshow("rectify_debug", debug_show)
        cv2.imshow("rectified_pass1", rectified1_show)
        cv2.imshow("rectified_pass2", pass2_show)
        cv2.imshow("rectified_pass2_refined", rectified2_show)
        cv2.imshow("rectified_refined_corner_cropped", refined_cropped_show)
        cv2.imshow("rectified_decoder", decoder_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
