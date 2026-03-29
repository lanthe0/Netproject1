"""WRGB 主协议编码入口。"""

from __future__ import annotations

import os
import sys
import time

import cv2

from _color2Dcode import encode_bin


def _write_color_video(frames: list, output_path: str, fps: int = 15) -> None:
    """将 WRGB 协议帧顺序写成视频。

    输入：
    - frames: `encode_bin()` 生成的彩色协议帧列表，元素为 BGR 图像。
    - output_path: 输出视频路径。
    - fps: 写出视频帧率，当前主链路默认使用 15fps。

    输出：
    - 无。函数直接在磁盘生成 mp4 文件。

    原理/流程：
    - WRGB 主协议编码后的结果已经是彩色 BGR 帧，而不是黑白 grid；
    - 因此这里不再复用旧的 `bin2video()`；
    - 直接创建 `cv2.VideoWriter` 顺序写出彩色视频，保持主入口改动最小。
    """

    if not frames:
        raise ValueError("没有可写入的视频帧")

    out_file = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    height, width = frames[0].shape[:2]

    writer = cv2.VideoWriter(
        out_file,
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频: {out_file}")

    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    """主编码入口。

    输入：
    - `sys.argv[1]`: 输入二进制文件路径。
    - `sys.argv[2]`: 输出视频路径。
    - `sys.argv[3]`: 视频最大时长，单位毫秒。
    - `sys.argv[4]`: 可选视频帧率；缺省时默认使用 30fps。

    输出：
    - 无。成功时在目标路径生成 WRGB 协议视频。

    原理/流程：
    - 先读取并校验命令行参数；
    - 再把时长上限折算成允许发送的最大逻辑帧数；
    - 调用 `_color2Dcode.encode_bin()` 生成 WRGB 协议帧；
    - 最后顺序写出彩色视频文件。
    """

    if len(sys.argv) not in (4, 5):
        print("Usage: python encode.py <input_file> <output_file> <max_length_ms> [fps]")
        print("Example: python encode.py test.bin output.mp4 1000")
        print("Example: python encode.py test.bin output.mp4 1000 15")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    try:
        max_length_of_video = int(sys.argv[3])
    except ValueError:
        print("Error: max_length_of_video 必须是整数（毫秒）")
        sys.exit(1)

    try:
        fps = int(sys.argv[4]) if len(sys.argv) == 5 else 30
    except ValueError:
        print("Error: fps 必须是整数")
        sys.exit(1)

    if fps <= 0:
        print("Error: fps 必须大于 0")
        sys.exit(1)

    if not os.path.exists(input_file_path):
        print(f"Error: 找不到输入文件 '{input_file_path}'")
        sys.exit(1)

    if os.path.getsize(input_file_path) > 10 * 1024 * 1024:
        print("输入文件大小超过 10MB")
        sys.exit(1)

    start_time = time.time()
    max_frames = max(1, int((max_length_of_video / 1000.0) * fps))

    print("--- 启动 WRGB 编码流水线 ---")
    print(f"输入文件: {input_file_path} ({os.path.getsize(input_file_path)} bytes)")
    print(f"目标路径: {output_file_path}")
    print(f"时长限制: {max_length_of_video} ms")
    print(f"目标帧率: {fps} fps")
    print(f"最大逻辑帧数: {max_frames}")

    try:
        frames = encode_bin(input_file_path, max_frames=max_frames)
        _write_color_video(frames, output_file_path, fps=fps)
        end_time = time.time()
        print("--- 编码成功 ---")
        print(f"总耗时: {end_time - start_time:.2f} 秒")
        print(f"生成视频位置: {os.path.abspath(output_file_path)}")
    except Exception as exc:
        print(f"发生错误: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
