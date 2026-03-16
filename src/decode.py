"""解码入口。

流程：
1. 从视频中提取并矫正二维码帧序列。
2. 将矫正后的帧序列交给现有协议解码器 `_2Dcode.decode_image`。
3. 输出还原后的二进制文件和逐位有效性标记文件。
"""

import os
import sys

import cv2
import numpy as np

try:
    from ._2Dcode import decode_image
    from .config import BIG_FINDER_SIZE, GRID_SIZE, QUIET_WIDTH, SMALL_FINDER_SIZE
except ImportError:
    from _2Dcode import decode_image
    from config import BIG_FINDER_SIZE, GRID_SIZE, QUIET_WIDTH, SMALL_FINDER_SIZE


WARPED_SIZE = GRID_SIZE * 10
FINDER_CENTER_OFFSET = QUIET_WIDTH + BIG_FINDER_SIZE / 2.0
FINDER_CENTER_SPAN = GRID_SIZE - 2 * FINDER_CENTER_OFFSET
SMALL_FINDER_CENTER = GRID_SIZE - QUIET_WIDTH - SMALL_FINDER_SIZE / 2.0


def extract_frames_from_video(video_path: str) -> list[np.ndarray]:
    """从视频文件中逐帧读取图像。"""
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
    """将 3 个大定位块中心点排序为：左上、右上、左下。"""
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
    """仅基于 3 个大定位块中心，估计二维码四个外角点。"""
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
    """在灰度图中搜索可能的定位块候选。"""
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
        area = cv2.contourArea(contour)
        if area < image_area * 0.0008:
            continue

        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), _ = rect
        if rw <= 1 or rh <= 1:
            continue

        aspect_ratio = max(rw, rh) / max(1.0, min(rw, rh))
        if aspect_ratio > 1.35:
            continue

        child = hierarchy[idx][2]
        nested_depth = 0
        while child != -1:
            nested_depth += 1
            child = hierarchy[child][2]
        if nested_depth < 2:
            continue

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

        score = area * nested_depth * black_ratio / aspect_ratio
        candidates.append((score, np.array([cx, cy], dtype=np.float32), area))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def locate_qr_corners(frame: np.ndarray) -> np.ndarray | None:
    """在单帧图像中定位二维码的四个外角点。"""
    if frame is None:
        return None

    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.astype(np.uint8, copy=False)

    candidates = _find_finder_candidates(gray)
    if len(candidates) < 3:
        return None

    top_candidates = candidates[:8]
    if len(top_candidates) >= 4:
        best_corners = None
        best_score = -1.0
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

        for i in range(len(top_candidates) - 3):
            for j in range(i + 1, len(top_candidates) - 2):
                for k in range(j + 1, len(top_candidates) - 1):
                    for m in range(k + 1, len(top_candidates)):
                        combo = [
                            top_candidates[i],
                            top_candidates[j],
                            top_candidates[k],
                            top_candidates[m],
                        ]
                        combo.sort(key=lambda item: item[2], reverse=True)
                        big = combo[:3]
                        small = combo[3]

                        big_centers = np.array(
                            [center for _, center, _ in big], dtype=np.float32
                        )
                        tl, tr, bl = _order_finder_centers(big_centers)
                        small_center = small[1]
                        if small_center[0] <= tl[0] or small_center[1] <= tl[1]:
                            continue

                        image_finder_centers = np.array(
                            [tl, tr, bl, small_center], dtype=np.float32
                        )
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

                        score = sum(item[0] for item in combo) + area / 1000.0
                        if score > best_score:
                            best_score = score
                            best_corners = corners

        if best_corners is not None:
            return best_corners

    centers = np.array([center for _, center, _ in top_candidates], dtype=np.float32)
    best_corners = None
    best_score = -1.0

    for i in range(len(centers) - 2):
        for j in range(i + 1, len(centers) - 1):
            for k in range(j + 1, len(centers)):
                sample = np.array([centers[i], centers[j], centers[k]], dtype=np.float32)
                tl, tr, bl = _order_finder_centers(sample)

                vec_x = tr - tl
                vec_y = bl - tl
                len_x = float(np.linalg.norm(vec_x))
                len_y = float(np.linalg.norm(vec_y))
                if len_x < 10 or len_y < 10:
                    continue

                scale_ratio = max(len_x, len_y) / max(1.0, min(len_x, len_y))
                if scale_ratio > 1.35:
                    continue

                cos_angle = abs(float(np.dot(vec_x, vec_y)) / max(len_x * len_y, 1e-6))
                if cos_angle > 0.25:
                    continue

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
    """根据二维码四角点对图像做透视矫正。"""
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
    return cv2.warpPerspective(
        frame,
        matrix,
        (WARPED_SIZE, WARPED_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def preprocess_frames_for_decode(frames: list[np.ndarray]) -> list[np.ndarray]:
    """对视频帧序列做“定位 + 透视矫正”预处理。"""
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


def main() -> None:
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
