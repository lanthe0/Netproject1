"""解码入口。

流程：
1. 从视频中提取并矫正二维码帧序列。
2. 将矫正后的帧序列交给现有协议解码器 `_2Dcode.decode_image`。
3. 输出还原后的二进制文件和逐位有效性标记文件。
"""

import os
import sys
import time

import numpy as np

try:
    # 包内导入，供编辑器/静态分析器正确解析。
    from ._2Dcode import decode_image
    from .utils.video_decode import video_to_qr_sequence
except ImportError:
    # 直接执行 `python src/decode.py ...` 时退回脚本模式导入。
    from _2Dcode import decode_image
    from utils.video_decode import video_to_qr_sequence


def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: python decode.py <input_video> <output_bin> <output_vbin>")
        sys.exit(1)

    # 命令行接口保持课程要求：输入视频、输出数据、输出有效性标记。
    input_video_path = sys.argv[1]
    output_bin_path = sys.argv[2]
    output_vbin_path = sys.argv[3]

    if not os.path.exists(input_video_path):
        print(f"Error: input video not found: {input_video_path}")
        sys.exit(1)

    start = time.time()
    try:
        # 先做“视频 -> 二维码序列”的前处理，再复用已有协议解码函数。
        qr_frames = video_to_qr_sequence(input_video_path)
        decode_image(np.asarray(qr_frames, dtype=object), output_bin_path, output_vbin_path)
    except Exception as exc:
        print(f"Decode failed: {exc}")
        sys.exit(1)

    duration = time.time() - start
    print(f"Decoded {len(qr_frames)} rectified frame(s) in {duration:.2f}s")
    print(f"Binary output: {os.path.abspath(output_bin_path)}")
    print(f"Validity output: {os.path.abspath(output_vbin_path)}")


if __name__ == "__main__":
    main()
