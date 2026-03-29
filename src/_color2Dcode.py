"""WRGB 主协议门面。

这个文件用于承接后续主链路对彩色协议的替换工作。
当前目标不是立刻改掉所有主入口，而是先提供与旧黑白 `_2Dcode.py`
一致的两个约定入口：

- `encode_bin(...)`
- `decode_image(...)`

这样上层 `encode.py / decode.py` 在切换主协议时，只需要改导入目标，
不用同时重写整套接口约定。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from _2Dcode import bytes_to_bits
from color_codec import WRGB_DATA_SIZE_LIMIT, ColorDecodeResult, decode_wrgb_frame, encode_wrgb_bitstream


def _normalize_frames(frames: Iterable[np.ndarray] | np.ndarray) -> list[np.ndarray]:
    """规整输入帧序列。

    输入：
    - frames: 可以是 `list[np.ndarray]`、生成器、单张 `np.ndarray`，
      或者旧主链路里常见的 `dtype=object` 数组。

    输出：
    - 统一规整后的帧列表。

    原理/流程：
    - 兼容旧 `_2Dcode.decode_image()` 的调用习惯；
    - 避免上层在切协议时，还要先修改传参形态；
    - 统一在这里做一次“单帧 / 多帧 / object 数组”的兜底归一化。
    """

    if isinstance(frames, np.ndarray):
        if frames.dtype == object:
            if frames.ndim == 2:
                return [np.asarray(frames)]
            return [np.asarray(frames[index]) for index in range(len(frames))]
        if frames.ndim == 2:
            return [frames]
        if frames.ndim >= 3:
            return [np.asarray(frames[index]) for index in range(len(frames))]
    return [np.asarray(frame) for frame in frames]


def _pack_bits(bits: list[int] | np.ndarray) -> bytes:
    """将 bit 序列打包成字节流。

    输入：
    - bits: 一维 bit 列表或 bit 数组。

    输出：
    - 按 8 位补零并打包后的 `bytes`。

    原理/流程：
    - 与旧协议输出 `out.bin / vout.bin` 的方式保持一致；
    - 若 bit 数不是 8 的倍数，则在尾部补 0；
    - 再调用 `np.packbits` 统一打包。
    """

    arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if arr.size == 0:
        return b""
    if arr.size % 8 != 0:
        arr = np.pad(arr, (0, 8 - arr.size % 8), mode="constant", constant_values=0)
    return np.packbits(arr).tobytes()


def encode_bin(
    path: str,
    *,
    max_frames: int = 0,
    pixel_per_cell: int = 10,
) -> list[np.ndarray]:
    """将二进制文件编码为 WRGB 协议帧序列。

    输入：
    - path: 输入二进制文件路径。
    - max_frames: 最多编码多少个逻辑帧；`0` 表示不截断。
    - pixel_per_cell: 输出协议帧中每个 cell 的放大像素倍数。

    输出：
    - WRGB 协议帧列表，列表元素为 OpenCV 使用的 BGR 图像。

    原理/流程：
    - 先读取文件并转换成 bit 流；
    - 若设置了 `max_frames`，则按 `WRGB_DATA_SIZE_LIMIT` 截断发送前缀；
    - 再调用 `color_codec.encode_wrgb_bitstream()` 生成完整 WRGB 协议帧。
    """

    file_path = Path(path)
    with file_path.open("rb") as file:
        raw_bytes = file.read()

    bits = bytes_to_bits(raw_bytes)
    if max_frames > 0:
        bits = bits[: max_frames * WRGB_DATA_SIZE_LIMIT]

    return encode_wrgb_bitstream(bits, pixel_per_cell=pixel_per_cell)


def decode_image(
    imgs: Iterable[np.ndarray] | np.ndarray,
    out_bin_path: str,
    out_vbin_path: str,
    *,
    sample_ratio: float = 0.4,
) -> None:
    """将 WRGB 协议帧序列解码成数据文件和有效位文件。

    输入：
    - imgs: 预处理后的 WRGB 帧序列；每一帧应为彩色 rectified 图。
    - out_bin_path: 输出数据文件路径。
    - out_vbin_path: 输出有效位图文件路径。
    - sample_ratio: WRGB 单元中心采样比例。

    输出：
    - 无。函数直接写出 `out.bin` 与 `vout.bin`。

    原理/流程：
    - 逐帧调用 `decode_wrgb_frame()` 读取 `frame_idx / bit_length / crc_ok / rs_ok / payload_bits`；
    - 对同一个 `frame_idx`，优先保留 `CRC` 成功的结果；
    - 按 `frame_idx` 排序后拼接 payload；
    - 若该帧 `CRC` 成功，则对应的 `vout` 位置写 1，否则写 0。
    """

    frames = _normalize_frames(imgs)
    decoded_frames: dict[int, tuple[np.ndarray, bool]] = {}

    for frame in frames:
        result: ColorDecodeResult = decode_wrgb_frame(frame, sample_ratio=sample_ratio)
        frame_idx = int(result.frame_idx)
        payload_bits = np.asarray(result.payload_bits[: int(result.bit_length)], dtype=np.uint8)
        valid = bool(result.crc_ok)

        previous = decoded_frames.get(frame_idx)
        if previous is None or (not previous[1] and valid):
            decoded_frames[frame_idx] = (payload_bits, valid)

    all_data_bits: list[int] = []
    all_valid_bits: list[int] = []

    for frame_idx in sorted(decoded_frames):
        payload_bits, valid = decoded_frames[frame_idx]
        payload_list = payload_bits.astype(np.uint8).tolist()
        all_data_bits.extend(payload_list)
        all_valid_bits.extend(([1] * len(payload_list)) if valid else ([0] * len(payload_list)))

    Path(out_bin_path).write_bytes(_pack_bits(all_data_bits))
    Path(out_vbin_path).write_bytes(_pack_bits(all_valid_bits))


__all__ = [
    "WRGB_DATA_SIZE_LIMIT",
    "decode_image",
    "encode_bin",
]
