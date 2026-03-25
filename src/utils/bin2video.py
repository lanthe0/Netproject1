"""将协议网格序列写成视频文件。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from config import GRID_SIZE
from utils.showimg import matrix_to_bw_image


UPSCALE = 10


def _grid_to_video_frame(grid: np.ndarray, upscale: int) -> np.ndarray:
    """把单帧协议网格转换成可直接写入视频的 BGR 图像。

    输入：
    - grid: 单帧二维码网格，元素为 0/1 或 bool。
    - upscale: 每个模块放大的像素倍数。

    输出：
    - `uint8` 类型的 BGR 图像，可直接送入 `cv2.VideoWriter.write()`。

    原理/流程：
    - 先把 0/1 网格转成灰度黑白图。
    - 再用最近邻插值放大，避免模块边缘被平滑。
    - 最后转成 BGR 三通道，兼容当前视频写入配置。
    """

    gray = matrix_to_bw_image(grid, pixel_per_cell=1)
    enlarged = cv2.resize(
        gray,
        (GRID_SIZE * upscale, GRID_SIZE * upscale),
        interpolation=cv2.INTER_NEAREST,
    )
    return cv2.cvtColor(enlarged, cv2.COLOR_GRAY2BGR)


def bin2video(
    grids: np.ndarray | Iterable[np.ndarray],
    output_path: str,
    max_length_of_video: int,
    fps: int = 15,
    upscale: int = UPSCALE,
) -> None:
    """将二维码网格序列流式写入 mp4 视频。

    输入：
    - grids: 二维码网格序列，可以是 `numpy.ndarray` 或任意可迭代对象。
    - output_path: 输出视频路径。
    - max_length_of_video: 允许的视频最大时长，单位毫秒。
    - fps: 输出视频帧率。
    - upscale: 每个网格模块放大的像素倍数。

    输出：
    - 无。成功时在 `output_path` 生成视频文件。

    原理/流程：
    - 先根据帧数校验时长约束。
    - 创建 `cv2.VideoWriter`。
    - 不再先构造完整 `img_group`，而是“生成一帧，立即写一帧”。
    - 这样可以减少中间图像列表的额外内存占用，也减少不必要的对象保留时间。
    """

    total_frames = len(grids) if hasattr(grids, "__len__") else None
    if total_frames is None:
        grids = list(grids)
        total_frames = len(grids)

    required_ms = (total_frames / fps) * 1000
    assert required_ms <= max_length_of_video, (
        f"当前时长限制 ({max_length_of_video}ms) 无法容纳数据，需要约 {int(required_ms)}ms"
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    frame_size = (GRID_SIZE * upscale, GRID_SIZE * upscale)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频写入器: {output_file}")

    try:
        for grid in grids:
            frame_bgr = _grid_to_video_frame(np.asarray(grid), upscale=upscale)
            writer.write(frame_bgr)
    finally:
        writer.release()

    print(f"成功生成视频: {output_file}, 共 {total_frames} 帧")
