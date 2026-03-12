import os
import sys

import cv2
import numpy as np

from _2Dcode import decode_image
from config import BIG_FINDER_SIZE, GRID_SIZE, QUIET_WIDTH, SMALL_FINDER_SIZE


WARPED_SIZE = GRID_SIZE * 10
# 大定位块中心距离二维码外边界的理论偏移量。
# 这里的几何关系来自编码端固定布局：
# 1. 二维码整体大小为 GRID_SIZE x GRID_SIZE。
# 2. 左上、右上、左下三个大定位块都从 QUIET_WIDTH 处开始绘制。
# 3. 大定位块边长为 BIG_FINDER_SIZE，因此其中心坐标为 QUIET_WIDTH + BIG_FINDER_SIZE / 2。
FINDER_CENTER_OFFSET = QUIET_WIDTH + BIG_FINDER_SIZE / 2.0
# 左上大定位块中心到右上/左下大定位块中心的理论跨度。
FINDER_CENTER_SPAN = GRID_SIZE - 2 * FINDER_CENTER_OFFSET
# 右下角小定位块中心在二维码本体坐标系中的理论位置。
SMALL_FINDER_CENTER = GRID_SIZE - QUIET_WIDTH - SMALL_FINDER_SIZE / 2.0


def extract_frames_from_video(video_path: str) -> list[np.ndarray]:
    """
    从视频文件中逐帧读取图像。

    这里不做抽样，也不做跳帧，目的是保持输入视频的完整帧序列，
    方便后续逐帧定位二维码并送入解码器。

    Args:
        video_path: 输入视频路径。

    Returns:
        视频中的全部帧，每一帧是一个 OpenCV 读取出的 numpy 数组。
    """
    frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("--- 视频信息 ---")
    print(f"视频路径：{video_path}")
    print(f"帧率：{fps}")
    print(f"总帧数：{frame_count}")
    print(f"分辨率：{width}x{height}")
    print("----------------")

    extracted_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        extracted_count += 1
        if frame_count > 0 and extracted_count % 10 == 0:
            progress = extracted_count / frame_count * 100
            print(f"已提取 {extracted_count}/{frame_count} 帧（{progress:.1f}%）")

    cap.release()
    print(f"视频帧提取完成，共 {len(frames)} 帧")
    return frames


def _order_finder_centers(centers: np.ndarray) -> np.ndarray:
    """
    将 3 个大定位块中心点排序为：左上、右上、左下。

    排序规则：
    1. x + y 最小的点视为左上。
    2. 剩余两个点里，x 更大的视为右上，另一个视为左下。

    Args:
        centers: 形状为 (3, 2) 的点集。

    Returns:
        排序后的 3 个点，顺序固定为 [左上, 右上, 左下]。
    """
    pts = np.asarray(centers, dtype=np.float32)
    sums = pts[:, 0] + pts[:, 1]
    tl_idx = int(np.argmin(sums))
    tl = pts[tl_idx]

    remaining = [idx for idx in range(3) if idx != tl_idx]
    p1, p2 = pts[remaining[0]], pts[remaining[1]]
    if p1[0] >= p2[0]:
        tr, bl = p1, p2
    else:
        tr, bl = p2, p1
    return np.array([tl, tr, bl], dtype=np.float32)


def _estimate_qr_corners(finder_centers: np.ndarray) -> np.ndarray | None:
    """
    仅基于 3 个大定位块中心，估计二维码四个外角点。

    这个函数是降级方案：
    当无法稳定检测到右下角小定位块时，仍然可以通过三个大定位块
    近似恢复二维码的外接四边形。

    注意：
    该估计更接近仿射恢复，在透视畸变很强时不如四点单应矩阵精确，
    但在正视或轻度拍摄角度偏差下通常足够工作。

    Args:
        finder_centers: 三个大定位块的中心点。

    Returns:
        估计得到的四个角点，顺序为 [左上, 右上, 右下, 左下]；
        若输入几何退化则返回 None。
    """
    tl, tr, bl = _order_finder_centers(finder_centers)

    vec_x = tr - tl
    vec_y = bl - tl
    norm_x = float(np.linalg.norm(vec_x))
    norm_y = float(np.linalg.norm(vec_y))
    if norm_x < 1e-6 or norm_y < 1e-6:
        return None

    ex = vec_x / FINDER_CENTER_SPAN
    ey = vec_y / FINDER_CENTER_SPAN

    tl_corner = tl - FINDER_CENTER_OFFSET * ex - FINDER_CENTER_OFFSET * ey
    tr_corner = tl_corner + GRID_SIZE * ex
    bl_corner = tl_corner + GRID_SIZE * ey
    br_corner = tl_corner + GRID_SIZE * ex + GRID_SIZE * ey
    return np.array([tl_corner, tr_corner, br_corner, bl_corner], dtype=np.float32)


def _find_finder_candidates(gray: np.ndarray) -> list[tuple[float, np.ndarray, float]]:
    """
    在灰度图中搜索可能的定位块候选。

    核心思路：
    1. 先用高斯滤波 + Otsu 二值化，得到黑白图。
    2. 使用 RETR_TREE 提取轮廓层级关系。
    3. 根据面积、长宽比、轮廓嵌套深度、区域黑色占比等条件，
       过滤出更像“定位块”的候选。

    之所以使用轮廓层级，是因为本项目的定位块不是纯实心方块，
    而是带嵌套结构的标记，和普通数据区小黑块在层级结构上不同。

    Args:
        gray: 输入灰度图。

    Returns:
        候选列表。每个元素包含：
        1. 候选评分 score
        2. 中心点坐标 center
        3. 原始轮廓面积 area
    """
    h, w = gray.shape
    image_area = float(h * w)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    candidates: list[tuple[float, np.ndarray, float]] = []

    for idx, contour in enumerate(contours):
        # 先过滤掉非常小的噪声轮廓。
        area = cv2.contourArea(contour)
        if area < image_area * 0.0008:
            continue

        # 使用最小外接旋转矩形来估计几何尺度和长宽比，
        # 这样比普通 axis-aligned bounding box 对旋转更稳。
        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), _ = rect
        if rw <= 1 or rh <= 1:
            continue

        # 定位块整体应接近正方形。
        aspect_ratio = max(rw, rh) / max(1.0, min(rw, rh))
        if aspect_ratio > 1.35:
            continue

        # 统计该轮廓沿着“子轮廓链”向下能走多深。
        # 深度越大，越像带层级结构的 Finder，而不是普通数据块。
        child = hierarchy[idx][2]
        nested_depth = 0
        while child != -1:
            nested_depth += 1
            child = hierarchy[child][2]
        if nested_depth < 2:
            continue

        # 过滤过于稀疏的轮廓。
        box_area = rw * rh
        fill_ratio = area / max(box_area, 1.0)
        if fill_ratio < 0.2:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + bw), min(h, y + bh)
        if x1 <= x0 or y1 <= y0:
            continue

        roi = binary[y0:y1, x0:x1]
        black_ratio = float(np.count_nonzero(roi)) / float(roi.size)
        if black_ratio < 0.2:
            continue

        # 综合评分：面积越大、层级越深、区域越像黑白定位块越优先。
        score = area * nested_depth * black_ratio / aspect_ratio
        candidates.append((score, np.array([cx, cy], dtype=np.float32), area))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def locate_qr_corners(frame: np.ndarray) -> np.ndarray | None:
    """
    在单帧图像中定位二维码的四个外角点。

    处理策略分两级：
    1. 优先尝试“四定位块”模式：
       同时使用左上、右上、左下三个大定位块以及右下小定位块，
       直接建立二维码坐标系到图像坐标系的单应矩阵。
       这是透视矫正最准确的方案。
    2. 若小定位块没有检测到，则退化到“三大定位块”模式：
       仅用三个大定位块估计外角点，保证解码流程尽量不中断。

    Args:
        frame: 输入图像，可以是灰度图，也可以是 BGR 彩色图。

    Returns:
        四个外角点，顺序为 [左上, 右上, 右下, 左下]。
        若定位失败则返回 None。
    """
    if frame is None:
        return None

    # 下游检测逻辑统一基于灰度图进行。
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.astype(np.uint8, copy=False)

    candidates = _find_finder_candidates(gray)
    if len(candidates) < 3:
        return None

    # 只保留得分最高的一小部分候选，减少组合爆炸。
    top_candidates = candidates[:8]
    if len(top_candidates) >= 4:
        best_corners = None
        best_score = -1.0
        # 这四个点是二维码“理论坐标系”中 4 个定位块中心的位置。
        code_finder_centers = np.array(
            [
                [FINDER_CENTER_OFFSET, FINDER_CENTER_OFFSET],
                [GRID_SIZE - FINDER_CENTER_OFFSET, FINDER_CENTER_OFFSET],
                [FINDER_CENTER_OFFSET, GRID_SIZE - FINDER_CENTER_OFFSET],
                [SMALL_FINDER_CENTER, SMALL_FINDER_CENTER],
            ],
            dtype=np.float32,
        )
        code_outer_corners = np.array(
            [[0, 0], [GRID_SIZE, 0], [GRID_SIZE, GRID_SIZE], [0, GRID_SIZE]],
            dtype=np.float32,
        )

        # 穷举 4 个候选块，尝试构成“3 大 1 小”的合法组合。
        for i in range(len(top_candidates) - 3):
            for j in range(i + 1, len(top_candidates) - 2):
                for k in range(j + 1, len(top_candidates) - 1):
                    for m in range(k + 1, len(top_candidates)):
                        combo = [top_candidates[i], top_candidates[j], top_candidates[k], top_candidates[m]]
                        # 面积更大的通常是三个大定位块，小定位块面积最小。
                        combo.sort(key=lambda item: item[2], reverse=True)
                        big = combo[:3]
                        small = combo[3]

                        big_centers = np.array([center for _, center, _ in big], dtype=np.float32)
                        tl, tr, bl = _order_finder_centers(big_centers)
                        small_center = small[1]
                        # 小定位块必须位于整体右下区域，否则当前组合无效。
                        if small_center[0] <= tl[0] or small_center[1] <= tl[1]:
                            continue

                        image_finder_centers = np.array(
                            [tl, tr, bl, small_center], dtype=np.float32
                        )
                        # 用理论定位点到图像定位点计算单应矩阵，
                        # 再把二维码四个理论外角投影到图像上。
                        homography = cv2.getPerspectiveTransform(
                            code_finder_centers, image_finder_centers
                        )
                        corners = cv2.perspectiveTransform(
                            code_outer_corners.reshape(1, -1, 2), homography
                        )[0]

                        contour = corners.astype(np.float32).reshape(-1, 1, 2)
                        area = abs(cv2.contourArea(contour))
                        if area < gray.shape[0] * gray.shape[1] * 0.01:
                            continue
                        if not cv2.isContourConvex(corners.astype(np.float32)):
                            continue

                        # 总评分由候选块得分和生成四边形面积共同决定。
                        score = sum(item[0] for item in combo) + area / 1000.0
                        if score > best_score:
                            best_score = score
                            best_corners = corners

        if best_corners is not None:
            return best_corners

    # 退化方案：只用三个大定位块估计四个角点。
    centers = np.array([center for _, center, _ in top_candidates], dtype=np.float32)
    best_corners = None
    best_score = -1.0

    for i in range(len(centers) - 2):
        for j in range(i + 1, len(centers) - 1):
            for k in range(j + 1, len(centers)):
                sample = np.array([centers[i], centers[j], centers[k]], dtype=np.float32)
                ordered = _order_finder_centers(sample)
                tl, tr, bl = ordered

                vec_x = tr - tl
                vec_y = bl - tl
                len_x = float(np.linalg.norm(vec_x))
                len_y = float(np.linalg.norm(vec_y))
                if len_x < 10 or len_y < 10:
                    continue

                # 二维码在正常投影下，横向和纵向尺度不会相差太离谱。
                scale_ratio = max(len_x, len_y) / max(1.0, min(len_x, len_y))
                if scale_ratio > 1.35:
                    continue

                # 三个大定位块应接近直角布局。
                cos_angle = abs(
                    float(np.dot(vec_x, vec_y)) / max(len_x * len_y, 1e-6)
                )
                if cos_angle > 0.25:
                    continue

                # 越接近正交、越接近等尺度、越大的组合越优先。
                score = min(len_x, len_y) / (1.0 + cos_angle + (scale_ratio - 1.0))
                if score <= best_score:
                    continue

                corners = _estimate_qr_corners(sample)
                if corners is None:
                    continue
                best_corners = corners
                best_score = score

    return best_corners


def rectify_qr_frame(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    根据二维码四角点对图像做透视矫正。

    矫正后的结果会被统一拉回到 WARPED_SIZE x WARPED_SIZE，
    这样 `_2Dcode.decode_image()` 就可以按固定模块尺寸做采样。

    Args:
        frame: 原始图像帧。
        corners: 二维码四个外角点，顺序为 [左上, 右上, 右下, 左下]。

    Returns:
        透视展开后的正视图。
    """
    dst = np.array(
        [
            [0, 0],
            [WARPED_SIZE - 1, 0],
            [WARPED_SIZE - 1, WARPED_SIZE - 1],
            [0, WARPED_SIZE - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(
        frame,
        matrix,
        (WARPED_SIZE, WARPED_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return warped


def preprocess_frames_for_decode(frames: list[np.ndarray]) -> list[np.ndarray]:
    """
    对视频帧序列做“定位 + 透视矫正”预处理。

    额外策略：
    1. 如果当前帧定位失败，但前一帧定位成功，则复用上一帧角点。
       这对轻微抖动或短暂模糊的视频更稳。
    2. 如果当前帧和上一帧都没有角点，则保留原帧直接下发给解码器。
       这样不会中断整段视频处理流程。

    Args:
        frames: 输入视频帧序列。

    Returns:
        预处理后的帧序列。成功定位的帧会被替换为矫正结果，
        失败的帧则保留原图。
    """
    processed_frames: list[np.ndarray] = []
    previous_corners: np.ndarray | None = None
    located = 0

    for idx, frame in enumerate(frames):
        corners = locate_qr_corners(frame)
        if corners is None and previous_corners is not None:
            corners = previous_corners

        if corners is None:
            processed_frames.append(frame)
            continue

        processed_frames.append(rectify_qr_frame(frame, corners))
        previous_corners = corners
        located += 1

        if (idx + 1) % 20 == 0:
            print(f"已完成 {idx + 1}/{len(frames)} 帧的二维码矫正")

    print(f"二维码定位成功：{located}/{len(frames)} 帧")
    return processed_frames


def main():
    """解码器主入口：提取视频帧、定位二维码、透视矫正并调用现有解码逻辑。"""
    if len(sys.argv) != 4:
        print("用法：python decode.py <输入视频> <输出二进制文件> <输出有效性文件>")
        print("示例：python decode.py output.mp4 output/decoded.bin output/vout.bin")
        sys.exit(1)

    input_video_path = sys.argv[1]
    output_bin_path = sys.argv[2]
    output_vbin_path = sys.argv[3]

    if not os.path.exists(input_video_path):
        print(f"错误：输入文件不存在：{input_video_path}")
        sys.exit(1)

    try:
        frames = extract_frames_from_video(input_video_path)
    except Exception as exc:
        print(f"错误：提取视频帧失败：{exc}")
        sys.exit(1)

    if not frames:
        print("错误：未从视频中提取到任何帧")
        sys.exit(1)

    try:
        prepared_frames = preprocess_frames_for_decode(frames)
    except Exception as exc:
        print(f"错误：二维码预处理失败：{exc}")
        sys.exit(1)

    print("\n--- 开始解码 ---")
    print(f"输入帧数：{len(prepared_frames)}")
    print(f"输出数据文件：{output_bin_path}")
    print(f"输出有效性文件：{output_vbin_path}")

    try:
        decode_image(prepared_frames, output_bin_path, output_vbin_path)
        print("\n--- 解码完成 ---")
        print("解码成功")
    except Exception as exc:
        print(f"错误：解码失败：{exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
