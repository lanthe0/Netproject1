"""颜色 payload 实验模块。

这个文件用于承载第一版颜色协议实验逻辑，尽量不直接污染现有黑白主链路。
当前只提供：

- 4 色 palette 定义
- 参考色块布局
- payload 的 2 bit / cell 编码
- 理想网格图上的颜色渲染与回读

后续如果确认颜色路线值得继续，再逐步接入 `_2Dcode.py`、`bin2video.py`
和 `video_decode.py`。
"""

from __future__ import annotations

import binascii
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import reedsolo

from _2Dcode import (
    BASE_GRID,
    DATA_ITER,
    ECC_BYTES,
    HEADER_SIZE,
    INFO_ITER,
    bytes_to_bits,
    calc_ecc_size,
    get_checkcode_from_bits,
    get_infoheader_from_bits,
)
from config import GRID_SIZE


ColorName = str


@dataclass(frozen=True)
class ColorCodecLayout:
    """颜色协议布局配置。

    输入：
    - ref_repeats: 每种参考色重复占用多少个 cell。

    输出：
    - 一个不可变布局对象，统一描述参考色块与 payload 的分配方式。

    原理/流程：
    - 颜色实验不改 finder 和 header。
    - 直接从 `DATA_ITER` 的起始位置划出一段作为颜色参考区。
    - 其余 cell 继续按现有蛇形顺序写 payload，只是每个 cell 从 1 bit 扩展到 2 bit。
    """

    ref_repeats: int = 4


@dataclass(frozen=True)
class ColorDecodeResult:
    """颜色协议单帧解码结果。

    输入：
    - 由 `decode_color_frame()` 在解码结束后构造。

    输出：
    - 包含 header、CRC、RS 和 payload 信息的结果对象。

    原理/流程：
    - 将颜色 payload 解码和黑白 header 解码结果统一打包。
    - 便于后续脚本直接读取 `frame_idx`、`crc_ok`、`payload_bits` 等关键信息。
    """

    frame_idx: int
    bit_length: int
    crc_ok: bool
    rs_ok: bool
    payload_bits: np.ndarray


@dataclass(frozen=True)
class WRGBDecision:
    """WRGB 单元判色结果。

    输入：
    - 由 `_classify_wrgb_patch_with_metrics()` 在单个 payload cell 判色结束后构造。

    输出：
    - symbol: 当前 cell 判定得到的 WRGB 类别，范围 `0..3`
    - confidence: 综合像素投票与均值距离后得到的置信度，范围 `[0, 1]`
    - vote_ratio: 像素投票中获胜类别所占比例
    - mean_margin: patch 均值在第一近邻和第二近邻之间的距离裕量

    原理/流程：
    - 给低置信单元邻域修正提供结构化输入；
    - 后处理阶段只修正低置信结果，避免把原本正确的高置信单元改坏。
    """

    symbol: int
    confidence: float
    vote_ratio: float
    mean_margin: float


PALETTE_A_RGB: dict[ColorName, tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "deep_cool": (40, 90, 210),
    "deep_warm": (210, 90, 40),
    "light_cool": (120, 230, 230),
    "light_warm": (240, 220, 120),
}

PALETTE_B_RGB: dict[ColorName, tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "deep_cool": (70, 110, 180),
    "deep_warm": (180, 110, 70),
    "light_cool": (150, 210, 210),
    "light_warm": (220, 205, 150),
}

PALETTE_WRGB_RGB: dict[ColorName, tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
}

REFERENCE_COLOR_ORDER: tuple[ColorName, ...] = (
    "black",
    "white",
    "deep_cool",
    "deep_warm",
    "light_cool",
    "light_warm",
)

SYMBOL_TO_COLOR: dict[int, ColorName] = {
    0b00: "deep_cool",
    0b01: "deep_warm",
    0b10: "light_cool",
    0b11: "light_warm",
}

COLOR_TO_SYMBOL: dict[ColorName, int] = {name: symbol for symbol, name in SYMBOL_TO_COLOR.items()}
COLOR_RS_CODEC = reedsolo.RSCodec(ECC_BYTES)

WRGB_REFERENCE_COLOR_ORDER: tuple[ColorName, ...] = (
    "white",
    "red",
    "green",
    "blue",
)

WRGB_SYMBOL_TO_COLOR: dict[int, ColorName] = {
    0b00: "white",
    0b01: "red",
    0b10: "green",
    0b11: "blue",
}

WRGB_COLOR_TO_SYMBOL: dict[ColorName, int] = {
    name: symbol for symbol, name in WRGB_SYMBOL_TO_COLOR.items()
}

WRGB_RS_CODEC = reedsolo.RSCodec(ECC_BYTES)


def _rgb_to_bgr(color_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """将 RGB 颜色元组转换成 OpenCV 使用的 BGR 顺序。"""

    r, g, b = color_rgb
    return b, g, r


def _normalize_palette(
    palette: dict[ColorName, tuple[int, int, int]] | None,
) -> dict[ColorName, tuple[int, int, int]]:
    """规范化颜色 palette。

    输入：
    - palette: 用户给定的 RGB palette；为空时默认使用方案 A。

    输出：
    - 完整的 RGB palette 字典。

    原理/流程：
    - 默认使用推荐的第一版 palette A。
    - 检查所有参考色和 payload 色是否都已定义，避免运行时缺色。
    """

    normalized = dict(PALETTE_A_RGB if palette is None else palette)
    required = set(REFERENCE_COLOR_ORDER)
    missing = sorted(required - set(normalized))
    if missing:
        raise ValueError(f"palette 缺少颜色定义: {missing}")
    return normalized


def _normalize_wrgb_palette(
    palette: dict[ColorName, tuple[int, int, int]] | None,
) -> dict[ColorName, tuple[int, int, int]]:
    """规范化 WRGB 实验方案使用的 palette。

    输入：
    - palette: 用户给定的 RGB palette；为空时默认使用纯 WRGB 方案。

    输出：
    - 至少包含 `black/white/red/green/blue` 的完整 palette。

    原理/流程：
    - WRGB 方案与 A/B 方案使用不同的颜色集合；
    - 因此单独校验所需颜色键，避免误把旧 palette 传进来。
    """

    normalized = dict(PALETTE_WRGB_RGB if palette is None else palette)
    required = {"black", "white", "red", "green", "blue"}
    missing = sorted(required - set(normalized))
    if missing:
        raise ValueError(f"WRGB palette 缺少颜色定义: {missing}")
    return normalized


def _build_wrgb_reference_blocks() -> dict[ColorName, list[tuple[int, int]]]:
    """构造 WRGB 方案的 4 个 4x4 参考块。

    输入：
    - 无，位置在协议网格中固定。

    输出：
    - 一个从颜色名到 cell 坐标列表的映射。

    原理/流程：
    - 在 payload 顶部区域预留 4 个 4x4 小块；
    - 分别写入 White / Red / Green / Blue；
    - 后续直接对这些块求中心颜色，用 RGB 欧氏距离做分类。
    """

    start_row = 2
    start_col = 27
    block_size = 4
    gap = 1
    blocks: dict[ColorName, list[tuple[int, int]]] = {}
    for index, color_name in enumerate(WRGB_REFERENCE_COLOR_ORDER):
        col0 = start_col + index * (block_size + gap)
        cells = [
            (row, col)
            for row in range(start_row, start_row + block_size)
            for col in range(col0, col0 + block_size)
        ]
        blocks[color_name] = cells
    return blocks


WRGB_REFERENCE_BLOCKS = _build_wrgb_reference_blocks()
WRGB_REFERENCE_CELLS = [
    cell
    for color_name in WRGB_REFERENCE_COLOR_ORDER
    for cell in WRGB_REFERENCE_BLOCKS[color_name]
]
WRGB_REFERENCE_SET = set(WRGB_REFERENCE_CELLS)
WRGB_PAYLOAD_CELLS = [cell for cell in DATA_ITER if cell not in WRGB_REFERENCE_SET]
WRGB_PAYLOAD_CELL_SET = set(WRGB_PAYLOAD_CELLS)

_DATA_ITER_SET = set(DATA_ITER)
for _cell in WRGB_REFERENCE_CELLS:
    if _cell not in _DATA_ITER_SET:
        raise ValueError(f"WRGB reference cell 不在 DATA_ITER 中: {_cell}")


def get_reference_cells(layout: ColorCodecLayout | None = None) -> list[tuple[int, int]]:
    """返回颜色参考区占用的 cell 列表。

    输入：
    - layout: 颜色布局配置；为空时使用默认布局。

    输出：
    - 参考色块对应的 cell 坐标列表，顺序与 `REFERENCE_COLOR_ORDER` 对应。

    原理/流程：
    - 直接从 `DATA_ITER` 的起始位置切出固定数量的 cell。
    - 每种颜色重复若干次，增加后续手机采样的稳定性。
    """

    actual_layout = layout or ColorCodecLayout()
    ref_count = len(REFERENCE_COLOR_ORDER) * actual_layout.ref_repeats
    return list(DATA_ITER[:ref_count])


def get_color_payload_cells(layout: ColorCodecLayout | None = None) -> list[tuple[int, int]]:
    """返回真正可写颜色 payload 的 cell 列表。

    输入：
    - layout: 颜色布局配置；为空时使用默认布局。

    输出：
    - 除去颜色参考区后的 payload cell 坐标列表。

    原理/流程：
    - 先从 `DATA_ITER` 中扣掉参考色块占用的前缀。
    - 其余 cell 全部留给颜色 payload 使用。
    """

    actual_layout = layout or ColorCodecLayout()
    ref_count = len(REFERENCE_COLOR_ORDER) * actual_layout.ref_repeats
    return list(DATA_ITER[ref_count:])


def get_color_payload_capacity_bits(layout: ColorCodecLayout | None = None) -> int:
    """计算颜色 payload 的理论容量。

    输入：
    - layout: 颜色布局配置；为空时使用默认布局。

    输出：
    - 在当前布局下可承载的 payload bit 数。

    原理/流程：
    - 第一版颜色协议约定每个 payload cell 写 2 bit。
    - 因此容量等于颜色 payload cell 数量乘以 2。
    """

    return len(get_color_payload_cells(layout)) * 2


def get_wrgb_reference_cells() -> list[tuple[int, int]]:
    """返回 WRGB 方案保留的参考块 cell 列表。"""

    return list(WRGB_REFERENCE_CELLS)


def get_wrgb_payload_cells() -> list[tuple[int, int]]:
    """返回 WRGB 方案真正可写 payload 的 cell 列表。"""

    return list(WRGB_PAYLOAD_CELLS)


def get_wrgb_payload_capacity_bits() -> int:
    """计算 WRGB 方案的理论 payload 容量。"""

    return len(WRGB_PAYLOAD_CELLS) * 2


def _calc_color_capacity_params(layout: ColorCodecLayout | None = None) -> tuple[int, int, int]:
    """计算完整颜色协议在当前布局下的容量参数。

    输入：
    - layout: 颜色布局配置；为空时使用默认布局。

    输出：
    - `(总颜色载荷 bit 容量, 最大原始数据字节数, 实际 ECC 开销字节数)`

    原理/流程：
    - 颜色 payload 区每个 cell 承载 2 bit。
    - 仍沿用现有 RS 分块编码模型，因此需要重新按颜色容量做一次二分搜索。
    """

    total_capacity_bits = get_color_payload_capacity_bits(layout)
    max_ecc_payload_bytes = total_capacity_bits // 8

    left, right = 0, max_ecc_payload_bytes
    while left < right:
        mid = (left + right + 1) // 2
        if calc_ecc_size(mid) <= max_ecc_payload_bytes:
            left = mid
        else:
            right = mid - 1

    max_data_bytes = left
    actual_ecc_bytes = calc_ecc_size(max_data_bytes) - max_data_bytes
    return total_capacity_bits, max_data_bytes, actual_ecc_bytes


COLOR_TOTAL_CAPACITY_BITS, COLOR_MAX_DATA_BYTES, COLOR_ACTUAL_ECC_BYTES = _calc_color_capacity_params()
COLOR_DATA_SIZE_LIMIT = COLOR_MAX_DATA_BYTES * 8


def _calc_wrgb_capacity_params() -> tuple[int, int, int]:
    """计算 WRGB 完整协议的容量参数。"""

    total_capacity_bits = get_wrgb_payload_capacity_bits()
    max_ecc_payload_bytes = total_capacity_bits // 8

    left, right = 0, max_ecc_payload_bytes
    while left < right:
        mid = (left + right + 1) // 2
        if calc_ecc_size(mid) <= max_ecc_payload_bytes:
            left = mid
        else:
            right = mid - 1

    max_data_bytes = left
    actual_ecc_bytes = calc_ecc_size(max_data_bytes) - max_data_bytes
    return total_capacity_bits, max_data_bytes, actual_ecc_bytes


WRGB_TOTAL_CAPACITY_BITS, WRGB_MAX_DATA_BYTES, WRGB_ACTUAL_ECC_BYTES = _calc_wrgb_capacity_params()
WRGB_DATA_SIZE_LIMIT = WRGB_MAX_DATA_BYTES * 8


def estimate_color_payload_capacity_bytes(
    *,
    seconds: float,
    fps: int = 15,
    layout: ColorCodecLayout | None = None,
) -> int:
    """估算给定时长下颜色 payload 能承载的原始字节数。

    输入：
    - seconds: 目标视频时长，单位为秒。
    - fps: 视频帧率。
    - layout: 颜色布局配置；为空时使用默认布局。

    输出：
    - 在当前颜色实验模型下，目标时长内可承载的近似字节数。

    原理/流程：
    - 先根据时长和帧率计算总帧数。
    - 再用每帧颜色 payload 理论容量乘以总帧数。
    - 最后按 8 bit = 1 byte 向下取整，给出一个适合做实验输入文件大小的估算值。
    """

    total_frames = max(1, int(round(seconds * fps)))
    return (get_color_payload_capacity_bits(layout) * total_frames) // 8


def _pair_bits(bits: np.ndarray) -> np.ndarray:
    """将 bit 流按 2 bit 一组打包成颜色 symbol。

    输入：
    - bits: 一维 bit 数组，元素取值为 0/1。

    输出：
    - 一维 symbol 数组，每个元素范围为 `0..3`。

    原理/流程：
    - 若 bit 数为奇数，则末尾补 0。
    - 每两个 bit 组合成一个 `2 bit` 符号，供单个颜色 cell 使用。
    """

    packed_bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if packed_bits.size % 2 != 0:
        packed_bits = np.pad(packed_bits, (0, 1), constant_values=0)
    paired = packed_bits.reshape(-1, 2)
    return ((paired[:, 0] << 1) | paired[:, 1]).astype(np.uint8)


def _bits_to_int(bits: np.ndarray) -> int:
    """将 bit 序列按大端顺序转换为整数。"""

    value = 0
    for bit in np.asarray(bits, dtype=np.uint8):
        value = (value << 1) | int(bit)
    return value


def _pack_bits(bits: Iterable[int]) -> bytes:
    """将 bit 序列打包为字节串。"""

    arr = np.asarray(list(bits), dtype=np.uint8)
    if arr.size == 0:
        return b""
    if arr.size % 8 != 0:
        arr = np.pad(arr, (0, 8 - arr.size % 8), constant_values=0)
    return np.packbits(arr).tobytes()


def _unpair_symbols(symbols: Iterable[int], bit_length: int) -> np.ndarray:
    """将颜色 symbol 还原成 bit 流。

    输入：
    - symbols: 颜色 symbol 序列，每个元素范围 `0..3`。
    - bit_length: 期望恢复出的原始 bit 长度。

    输出：
    - 一维 bit 数组。

    原理/流程：
    - 每个 symbol 展开成两个 bit。
    - 最后按调用方给定的原始 bit 长度裁剪补零尾部。
    """

    symbol_list = [int(symbol) for symbol in symbols]
    out = np.zeros(len(symbol_list) * 2, dtype=np.uint8)
    for idx, symbol in enumerate(symbol_list):
        out[idx * 2] = (int(symbol) >> 1) & 1
        out[idx * 2 + 1] = int(symbol) & 1
    return out[:bit_length]


def render_color_frame(
    bits: np.ndarray,
    *,
    layout: ColorCodecLayout | None = None,
    palette: dict[ColorName, tuple[int, int, int]] | None = None,
    pixel_per_cell: int = 10,
) -> np.ndarray:
    """将 payload bit 流渲染成带颜色的协议帧。

    输入：
    - bits: 待写入 payload 的 bit 流。
    - layout: 颜色布局配置。
    - palette: RGB 颜色表；为空时使用方案 A。
    - pixel_per_cell: 每个 cell 放大的像素倍数。

    输出：
    - `uint8` 类型的 BGR 彩色图像，可直接保存或写入视频。

    原理/流程：
    - 先从黑白 `BASE_GRID` 生成基础白底/黑结构层。
    - 再在预留的参考色块位置写入已知颜色。
    - 最后将 payload 按 `2 bit / cell` 映射成 4 色，写入 payload 区。
    - 输出图像保持彩色，方便后续做颜色采样实验。
    """

    actual_layout = layout or ColorCodecLayout()
    actual_palette = _normalize_palette(palette)
    payload_cells = get_color_payload_cells(actual_layout)
    ref_cells = get_reference_cells(actual_layout)

    symbols = _pair_bits(np.asarray(bits, dtype=np.uint8))
    if len(symbols) > len(payload_cells):
        raise ValueError(
            f"颜色 payload 长度 {len(symbols)} 超过容量 {len(payload_cells)} 个 cell"
        )

    frame = np.full((GRID_SIZE, GRID_SIZE, 3), 255, dtype=np.uint8)
    black_mask = BASE_GRID.astype(bool)
    frame[black_mask] = _rgb_to_bgr(actual_palette["black"])

    repeats = actual_layout.ref_repeats
    for color_index, color_name in enumerate(REFERENCE_COLOR_ORDER):
        start = color_index * repeats
        end = start + repeats
        for row, col in ref_cells[start:end]:
            frame[row, col] = _rgb_to_bgr(actual_palette[color_name])

    for symbol, (row, col) in zip(symbols, payload_cells):
        frame[row, col] = _rgb_to_bgr(actual_palette[SYMBOL_TO_COLOR[int(symbol)]])

    if pixel_per_cell <= 1:
        return frame

    return cv2.resize(
        frame,
        (GRID_SIZE * pixel_per_cell, GRID_SIZE * pixel_per_cell),
        interpolation=cv2.INTER_NEAREST,
    )


def render_wrgb_frame(
    bits: np.ndarray,
    *,
    palette: dict[ColorName, tuple[int, int, int]] | None = None,
    pixel_per_cell: int = 10,
) -> np.ndarray:
    """将 WRGB 方案的 payload bit 流渲染成协议帧。

    输入：
    - bits: 待写入 payload 的 bit 流。
    - palette: WRGB 实验 palette；为空时使用默认纯 WRGB。
    - pixel_per_cell: 每个 cell 放大的像素倍数。

    输出：
    - 一张带 WRGB 参考块与彩色 payload 的 BGR 图像。

    原理/流程：
    - 顶部固定放置 4 个 4x4 的 W/R/G/B 参考块；
    - 其余 payload cell 继续按 `1 cell = 2 bit` 写入；
    - 颜色分类时直接依赖这 4 个参考块做 RGB 欧氏距离匹配。
    """

    actual_palette = _normalize_wrgb_palette(palette)
    payload_cells = get_wrgb_payload_cells()

    symbols = _pair_bits(np.asarray(bits, dtype=np.uint8))
    if len(symbols) > len(payload_cells):
        raise ValueError(
            f"WRGB payload 长度 {len(symbols)} 超过容量 {len(payload_cells)} 个 cell"
        )

    frame = np.full((GRID_SIZE, GRID_SIZE, 3), 255, dtype=np.uint8)
    black_mask = BASE_GRID.astype(bool)
    frame[black_mask] = _rgb_to_bgr(actual_palette["black"])

    for color_name in WRGB_REFERENCE_COLOR_ORDER:
        for row, col in WRGB_REFERENCE_BLOCKS[color_name]:
            frame[row, col] = _rgb_to_bgr(actual_palette[color_name])

    for symbol, (row, col) in zip(symbols, payload_cells):
        frame[row, col] = _rgb_to_bgr(actual_palette[WRGB_SYMBOL_TO_COLOR[int(symbol)]])

    if pixel_per_cell <= 1:
        return frame

    return cv2.resize(
        frame,
        (GRID_SIZE * pixel_per_cell, GRID_SIZE * pixel_per_cell),
        interpolation=cv2.INTER_NEAREST,
    )


def _write_binary_cells_on_frame(
    frame_bgr: np.ndarray,
    cells: list[tuple[int, int]],
    bits: np.ndarray,
) -> None:
    """在网格级彩色帧上写入黑白二值模块。

    输入：
    - frame_bgr: `GRID_SIZE x GRID_SIZE x 3` 的彩色网格帧。
    - cells: 需要写入的 cell 坐标序列。
    - bits: 与 `cells` 对齐的 bit 序列，`1` 为黑，`0` 为白。

    输出：
    - 无，直接原地修改 `frame_bgr`。

    原理/流程：
    - 颜色协议中 finder 和 header 继续保持黑白。
    - 因此 header 直接在彩色帧上覆写成黑白 cell，而不是通过颜色 symbol 表示。
    """

    for (row, col), bit in zip(cells, np.asarray(bits, dtype=np.uint8)):
        frame_bgr[row, col] = (0, 0, 0) if int(bit) == 1 else (255, 255, 255)


def _sample_binary_cells(
    frame_bgr: np.ndarray,
    cells: list[tuple[int, int]],
    *,
    sample_ratio: float = 0.5,
) -> np.ndarray:
    """从彩色协议帧中读取黑白二值模块。

    输入：
    - frame_bgr: 已对齐的 BGR 彩色帧。
    - cells: 需要读取的黑白 cell 坐标列表。
    - sample_ratio: 中心采样比例。

    输出：
    - 二值 bit 数组，`1` 表示黑模块。

    原理/流程：
    - 先对齐到协议网格尺寸。
    - 再对每个 cell 的中心窗口取平均灰度。
    - 平均灰度小于 128 视为黑色。
    """

    normalized_frame, module = _frame_to_grid_bgr(frame_bgr)
    bits = np.zeros(len(cells), dtype=np.uint8)
    for index, (row, col) in enumerate(cells):
        sample_bgr = _sample_module_mean(normalized_frame, row, col, module, sample_ratio)
        gray = float(sample_bgr.mean())
        bits[index] = 1 if gray < 128.0 else 0
    return bits


def encode_color_payload_frame(
    bits: np.ndarray,
    frame_idx: int,
    *,
    layout: ColorCodecLayout | None = None,
    palette: dict[ColorName, tuple[int, int, int]] | None = None,
    pixel_per_cell: int = 10,
) -> np.ndarray:
    """将单帧原始 payload bit 编码成完整颜色协议帧。

    输入：
    - bits: 单帧原始 payload bit 流。
    - frame_idx: 当前帧编号，将写入黑白 header。
    - layout/palette/pixel_per_cell: 颜色协议渲染参数。

    输出：
    - 一张带黑白 header、RS 和颜色 payload 的 BGR 彩色帧。

    原理/流程：
    - 先对原始 payload 做 RS 编码，得到要写入颜色 cell 的 ECC bit 流。
    - 再调用 `render_color_frame()` 把 ECC bit 写入颜色 payload 区。
    - 最后把 `CRC32 + bit_len + frame_idx` 作为黑白 header 覆写到 `INFO_ITER` 区域。
    """

    payload_bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if payload_bits.size > COLOR_DATA_SIZE_LIMIT:
        raise ValueError(
            f"颜色协议单帧 payload 长度 {payload_bits.size} 超过上限 {COLOR_DATA_SIZE_LIMIT}"
        )

    raw_bytes = _pack_bits(payload_bits)
    encoded_payload_bytes = COLOR_RS_CODEC.encode(raw_bytes)
    encoded_payload_bits = bytes_to_bits(encoded_payload_bytes)
    if encoded_payload_bits.size > get_color_payload_capacity_bits(layout):
        raise ValueError("颜色 payload 加上 RS 后超过颜色协议容量")

    frame = render_color_frame(
        encoded_payload_bits,
        layout=layout,
        palette=palette,
        pixel_per_cell=1,
    )
    header_bits = np.concatenate(
        (
            get_checkcode_from_bits(raw_bytes),
            get_infoheader_from_bits(int(payload_bits.size), int(frame_idx)),
        )
    )
    _write_binary_cells_on_frame(frame, list(INFO_ITER[:HEADER_SIZE]), header_bits)

    if pixel_per_cell <= 1:
        return frame

    return cv2.resize(
        frame,
        (GRID_SIZE * pixel_per_cell, GRID_SIZE * pixel_per_cell),
        interpolation=cv2.INTER_NEAREST,
    )


def encode_wrgb_payload_frame(
    bits: np.ndarray,
    frame_idx: int,
    *,
    palette: dict[ColorName, tuple[int, int, int]] | None = None,
    pixel_per_cell: int = 10,
) -> np.ndarray:
    """将单帧原始 payload 编码成完整 WRGB 协议帧。"""

    payload_bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if payload_bits.size > WRGB_DATA_SIZE_LIMIT:
        raise ValueError(
            f"WRGB 协议单帧 payload 长度 {payload_bits.size} 超过上限 {WRGB_DATA_SIZE_LIMIT}"
        )

    raw_bytes = _pack_bits(payload_bits)
    encoded_payload_bytes = WRGB_RS_CODEC.encode(raw_bytes)
    encoded_payload_bits = bytes_to_bits(encoded_payload_bytes)
    if encoded_payload_bits.size > get_wrgb_payload_capacity_bits():
        raise ValueError("WRGB payload 加上 RS 后超过协议容量")

    frame = render_wrgb_frame(
        encoded_payload_bits,
        palette=palette,
        pixel_per_cell=1,
    )
    header_bits = np.concatenate(
        (
            get_checkcode_from_bits(raw_bytes),
            get_infoheader_from_bits(int(payload_bits.size), int(frame_idx)),
        )
    )
    _write_binary_cells_on_frame(frame, list(INFO_ITER[:HEADER_SIZE]), header_bits)

    if pixel_per_cell <= 1:
        return frame

    return cv2.resize(
        frame,
        (GRID_SIZE * pixel_per_cell, GRID_SIZE * pixel_per_cell),
        interpolation=cv2.INTER_NEAREST,
    )


def decode_color_frame(
    frame_bgr: np.ndarray,
    *,
    layout: ColorCodecLayout | None = None,
    palette: dict[ColorName, tuple[int, int, int]] | None = None,
    sample_ratio: float = 0.5,
) -> ColorDecodeResult:
    """解码单张完整颜色协议帧。

    输入：
    - frame_bgr: 已对齐或待读取的 BGR 彩色协议帧。
    - layout/palette/sample_ratio: 颜色解码参数。

    输出：
    - `ColorDecodeResult`，包含 `frame_idx`、`bit_length`、`crc_ok`、`rs_ok` 和 payload bit。

    原理/流程：
    - 先从黑白 header 中读取 `CRC + bit_len + frame_idx`。
    - 再根据 bit 长度推导 RS 编码后的颜色 payload 长度。
    - 用颜色参考块恢复颜色中心，并从颜色 payload 区回读 ECC bit。
    - 最后执行 RS 解码并做 CRC 校验。
    """

    actual_layout = layout or ColorCodecLayout()
    actual_palette = _normalize_palette(palette)
    header_bits = _sample_binary_cells(frame_bgr, list(INFO_ITER[:HEADER_SIZE]), sample_ratio=sample_ratio)

    expected_crc = _bits_to_int(header_bits[:32])
    info_bits = header_bits[32:64]
    bit_length = _bits_to_int(info_bits[:16])
    frame_idx = _bits_to_int(info_bits[16:32])

    if bit_length > COLOR_DATA_SIZE_LIMIT:
        bit_length = COLOR_DATA_SIZE_LIMIT

    byte_length = (bit_length + 7) // 8
    encoded_byte_length = calc_ecc_size(byte_length)
    encoded_bit_length = encoded_byte_length * 8
    encoded_payload_bits = decode_color_bits(
        frame_bgr,
        encoded_bit_length,
        layout=actual_layout,
        palette=actual_palette,
        sample_ratio=sample_ratio,
    )

    rs_ok = False
    crc_ok = False
    payload_bits = np.zeros(bit_length, dtype=np.uint8)
    try:
        corrected_raw_bytes, _, _ = COLOR_RS_CODEC.decode(_pack_bits(encoded_payload_bits))
        corrected_bytes = bytes(corrected_raw_bytes)
        rs_ok = True
        payload_bits = bytes_to_bits(corrected_bytes)[:bit_length]
        actual_crc = binascii.crc32(corrected_bytes[:byte_length]) & 0xFFFFFFFF
        crc_ok = actual_crc == expected_crc
    except reedsolo.ReedSolomonError:
        pass

    return ColorDecodeResult(
        frame_idx=frame_idx,
        bit_length=bit_length,
        crc_ok=crc_ok,
        rs_ok=rs_ok,
        payload_bits=payload_bits,
    )


def decode_wrgb_bits(
    frame_bgr: np.ndarray,
    bit_length: int,
    *,
    palette: dict[ColorName, tuple[int, int, int]] | None = None,
    sample_ratio: float = 0.4,
) -> np.ndarray:
    """从对齐后的 WRGB 协议帧中恢复编码后 payload bit。

    输入：
    - frame_bgr: 已与协议网格对齐的 BGR 彩色帧。
    - bit_length: 期望恢复的编码后 payload bit 数。
    - palette: WRGB 方案的参考 palette。
    - sample_ratio: 中心采样窗口比例。

    输出：
    - 采样恢复得到的编码后 payload bit 数组。

    原理/流程：
    - 先从 4 个 4x4 的 WRGB 参考块估计每帧实际参考色；
    - 再用白色参考块估计通道增益，先做一轮轻量白平衡；
    - 将候选颜色转到 Lab 空间，用颜色距离替代直接 RGB 欧氏距离；
    - 对每个 payload cell 同时做像素级投票和均值分类，再联合决策。
    """

    actual_palette = _normalize_wrgb_palette(palette)
    normalized_frame, module = _frame_to_grid_bgr(frame_bgr)
    _reference_centers, symbol_centers_lab, channel_gains = _estimate_wrgb_symbol_centers(
        normalized_frame,
        module,
        actual_palette,
        sample_ratio,
    )

    required_symbols = (bit_length + 1) // 2
    decisions: list[WRGBDecision] = []
    for row, col in WRGB_PAYLOAD_CELLS[:required_symbols]:
        patch = _sample_module_patch(normalized_frame, row, col, module, sample_ratio)
        decision = _classify_wrgb_patch_with_metrics(
            patch,
            symbol_centers_lab,
            channel_gains,
        )
        decisions.append(decision)

    symbol_map, confidence_map = _build_wrgb_symbol_maps(required_symbols, decisions)
    decoded_symbols = _apply_wrgb_low_confidence_correction(
        symbol_map,
        confidence_map,
        required_symbols=required_symbols,
    )

    return _unpair_symbols(decoded_symbols, bit_length)


def decode_wrgb_frame(
    frame_bgr: np.ndarray,
    *,
    palette: dict[ColorName, tuple[int, int, int]] | None = None,
    sample_ratio: float = 0.4,
) -> ColorDecodeResult:
    """解码单张完整 WRGB 协议帧。"""

    _ = _normalize_wrgb_palette(palette)
    header_bits = _sample_binary_cells(frame_bgr, list(INFO_ITER[:HEADER_SIZE]), sample_ratio=sample_ratio)

    expected_crc = _bits_to_int(header_bits[:32])
    info_bits = header_bits[32:64]
    bit_length = _bits_to_int(info_bits[:16])
    frame_idx = _bits_to_int(info_bits[16:32])

    if bit_length > WRGB_DATA_SIZE_LIMIT:
        bit_length = WRGB_DATA_SIZE_LIMIT

    byte_length = (bit_length + 7) // 8
    encoded_byte_length = calc_ecc_size(byte_length)
    encoded_bit_length = encoded_byte_length * 8
    encoded_payload_bits = decode_wrgb_bits(
        frame_bgr,
        encoded_bit_length,
        palette=palette,
        sample_ratio=sample_ratio,
    )

    rs_ok = False
    crc_ok = False
    payload_bits = np.zeros(bit_length, dtype=np.uint8)
    try:
        corrected_raw_bytes, _, _ = WRGB_RS_CODEC.decode(_pack_bits(encoded_payload_bits))
        corrected_bytes = bytes(corrected_raw_bytes)
        rs_ok = True
        payload_bits = bytes_to_bits(corrected_bytes)[:bit_length]
        actual_crc = binascii.crc32(corrected_bytes[:byte_length]) & 0xFFFFFFFF
        crc_ok = actual_crc == expected_crc
    except reedsolo.ReedSolomonError:
        pass

    return ColorDecodeResult(
        frame_idx=frame_idx,
        bit_length=bit_length,
        crc_ok=crc_ok,
        rs_ok=rs_ok,
        payload_bits=payload_bits,
    )


def encode_color_bitstream(
    bits: np.ndarray,
    *,
    layout: ColorCodecLayout | None = None,
    palette: dict[ColorName, tuple[int, int, int]] | None = None,
    pixel_per_cell: int = 10,
) -> list[np.ndarray]:
    """将连续 bit 流切分成多张完整颜色协议帧。

    输入：
    - bits: 连续原始 bit 流。
    - layout/palette/pixel_per_cell: 颜色协议渲染参数。

    输出：
    - 完整颜色协议帧列表，每帧都含 header、frame_idx 和 RS。

    原理/流程：
    - 先按 `COLOR_DATA_SIZE_LIMIT` 切分原始 bit。
    - 再逐帧调用 `encode_color_payload_frame()`，写入各自的 `frame_idx`。
    """

    actual_bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    frames: list[np.ndarray] = []
    for frame_idx, start in enumerate(range(0, len(actual_bits), COLOR_DATA_SIZE_LIMIT)):
        chunk = actual_bits[start : start + COLOR_DATA_SIZE_LIMIT]
        frames.append(
            encode_color_payload_frame(
                chunk,
                frame_idx,
                layout=layout,
                palette=palette,
                pixel_per_cell=pixel_per_cell,
            )
        )
    return frames


def encode_wrgb_bitstream(
    bits: np.ndarray,
    *,
    palette: dict[ColorName, tuple[int, int, int]] | None = None,
    pixel_per_cell: int = 10,
) -> list[np.ndarray]:
    """将连续 bit 流编码成多张完整 WRGB 协议帧。"""

    actual_bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    frames: list[np.ndarray] = []
    for frame_idx, start in enumerate(range(0, len(actual_bits), WRGB_DATA_SIZE_LIMIT)):
        chunk = actual_bits[start : start + WRGB_DATA_SIZE_LIMIT]
        frames.append(
            encode_wrgb_payload_frame(
                chunk,
                frame_idx,
                palette=palette,
                pixel_per_cell=pixel_per_cell,
            )
        )
    return frames


def encode_color_bin(
    path: str,
    *,
    layout: ColorCodecLayout | None = None,
    palette: dict[ColorName, tuple[int, int, int]] | None = None,
    pixel_per_cell: int = 10,
) -> list[np.ndarray]:
    """将二进制文件编码成多张完整颜色协议帧。

    输入：
    - path: 输入二进制文件路径。
    - layout/palette/pixel_per_cell: 颜色协议渲染参数。

    输出：
    - 完整颜色协议帧列表。

    原理/流程：
    - 先读取文件内容并转成 bit 流。
    - 再调用 `encode_color_bitstream()` 完成多帧颜色协议编码。
    """

    with open(path, "rb") as file:
        data = file.read()
    return encode_color_bitstream(
        bytes_to_bits(data),
        layout=layout,
        palette=palette,
        pixel_per_cell=pixel_per_cell,
    )


def save_color_frame(
    bits: np.ndarray,
    output_path: str,
    *,
    layout: ColorCodecLayout | None = None,
    palette: dict[ColorName, tuple[int, int, int]] | None = None,
    pixel_per_cell: int = 10,
) -> Path:
    """将单帧颜色协议图保存到磁盘。

    输入：
    - bits: 单帧 payload bit 流。
    - output_path: 输出图片路径。
    - layout/palette/pixel_per_cell: 颜色渲染相关参数。

    输出：
    - 实际保存的输出路径对象。

    原理/流程：
    - 先调用 `render_color_frame()` 生成彩色协议帧。
    - 再创建目录并写成图片文件，便于肉眼检查布局和颜色分布。
    """

    frame = render_color_frame(
        bits,
        layout=layout,
        palette=palette,
        pixel_per_cell=pixel_per_cell,
    )
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_file), frame)
    if not ok:
        raise RuntimeError(f"无法保存颜色测试帧: {out_file}")
    return out_file


def render_color_frames_from_bits(
    bits: np.ndarray,
    *,
    layout: ColorCodecLayout | None = None,
    palette: dict[ColorName, tuple[int, int, int]] | None = None,
    pixel_per_cell: int = 10,
) -> list[np.ndarray]:
    """将整段 bit 流切成多帧颜色协议图。

    输入：
    - bits: 连续 payload bit 流。
    - layout/palette/pixel_per_cell: 颜色渲染相关参数。

    输出：
    - 彩色协议帧列表，每帧都是 BGR 图像。

    原理/流程：
    - 按单帧颜色 payload 理论容量切分 bit 流。
    - 对每一段调用 `render_color_frame()`。
    - 不额外引入 header/RS，只服务于第一版颜色实验。
    """

    actual_bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    frame_bit_capacity = get_color_payload_capacity_bits(layout)
    frames: list[np.ndarray] = []
    for start in range(0, len(actual_bits), frame_bit_capacity):
        chunk = actual_bits[start : start + frame_bit_capacity]
        frames.append(
            render_color_frame(
                chunk,
                layout=layout,
                palette=palette,
                pixel_per_cell=pixel_per_cell,
            )
        )
    return frames


def save_color_video(
    bits: np.ndarray,
    output_path: str,
    *,
    fps: int = 15,
    layout: ColorCodecLayout | None = None,
    palette: dict[ColorName, tuple[int, int, int]] | None = None,
    pixel_per_cell: int = 10,
) -> Path:
    """将 bit 流保存成一段颜色实验视频。

    输入：
    - bits: 连续 payload bit 流。
    - output_path: 输出视频路径。
    - fps: 视频帧率。
    - layout/palette/pixel_per_cell: 颜色渲染相关参数。

    输出：
    - 实际保存的视频路径对象。

    原理/流程：
    - 先将 bit 流切成多帧彩色协议图。
    - 再用 OpenCV `VideoWriter` 顺序写出 mp4 视频。
    - 只用于颜色实验，不影响现有黑白视频生成逻辑。
    """

    frames = render_color_frames_from_bits(
        bits,
        layout=layout,
        palette=palette,
        pixel_per_cell=pixel_per_cell,
    )
    if not frames:
        raise ValueError("颜色视频至少需要一帧数据")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_file),
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建颜色测试视频: {out_file}")

    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

    return out_file


def render_color_protocol_frames_from_bits(
    bits: np.ndarray,
    *,
    layout: ColorCodecLayout | None = None,
    palette: dict[ColorName, tuple[int, int, int]] | None = None,
    pixel_per_cell: int = 10,
) -> list[np.ndarray]:
    """将连续 bit 流切分成多张完整颜色协议帧。

    输入：
    - bits: 连续原始 bit 流。
    - layout/palette/pixel_per_cell: 颜色协议渲染参数。

    输出：
    - 带 header、frame_idx 和 RS 的完整颜色协议帧列表。

    原理/流程：
    - 调用 `encode_color_bitstream()` 生成完整协议帧。
    - 这个接口专门服务于后续视频播放和手机拍摄测试。
    """

    return encode_color_bitstream(
        bits,
        layout=layout,
        palette=palette,
        pixel_per_cell=pixel_per_cell,
    )


def save_color_protocol_video(
    bits: np.ndarray,
    output_path: str,
    *,
    fps: int = 15,
    layout: ColorCodecLayout | None = None,
    palette: dict[ColorName, tuple[int, int, int]] | None = None,
    pixel_per_cell: int = 10,
) -> Path:
    """将连续 bit 流保存为完整颜色协议视频。

    输入：
    - bits: 连续原始 bit 流。
    - output_path: 输出视频路径。
    - fps: 视频帧率。
    - layout/palette/pixel_per_cell: 颜色协议渲染参数。

    输出：
    - 实际保存的视频路径对象。

    原理/流程：
    - 先将 bit 流编码成完整颜色协议帧。
    - 每帧都包含黑白 header、frame_idx 和 RS。
    - 再顺序写出视频，供手机拍摄和多帧测试使用。
    """

    frames = render_color_protocol_frames_from_bits(
        bits,
        layout=layout,
        palette=palette,
        pixel_per_cell=pixel_per_cell,
    )
    if not frames:
        raise ValueError("完整颜色协议视频至少需要一帧数据")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_file),
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建完整颜色协议视频: {out_file}")

    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

    return out_file


def save_wrgb_protocol_video(
    bits: np.ndarray,
    output_path: str,
    *,
    fps: int = 15,
    palette: dict[ColorName, tuple[int, int, int]] | None = None,
    pixel_per_cell: int = 10,
) -> Path:
    """将连续 bit 流保存为完整 WRGB 协议视频。"""

    frames = encode_wrgb_bitstream(
        bits,
        palette=palette,
        pixel_per_cell=pixel_per_cell,
    )
    if not frames:
        raise ValueError("完整 WRGB 协议视频至少需要一帧数据")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_file),
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建完整 WRGB 协议视频: {out_file}")

    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

    return out_file


def _sample_module_mean(
    frame_bgr: np.ndarray,
    row: int,
    col: int,
    module: int,
    sample_ratio: float,
) -> np.ndarray:
    """采样单个 cell 中心区域的平均颜色。

    输入：
    - frame_bgr: 已对齐到协议网格的 BGR 图像。
    - row/col: cell 坐标。
    - module: 单个 cell 的像素边长。
    - sample_ratio: 中心采样窗口占 cell 边长的比例。

    输出：
    - 长度为 3 的 BGR 平均颜色向量。

    原理/流程：
    - 不直接对整块均值采样，避免边缘串色和几何变形影响。
    - 只取 cell 中央窗口，再求均值。
    """

    patch = _sample_module_patch(frame_bgr, row, col, module, sample_ratio)
    return patch.reshape(-1, 3).mean(axis=0)


def _sample_module_patch(
    frame_bgr: np.ndarray,
    row: int,
    col: int,
    module: int,
    sample_ratio: float,
) -> np.ndarray:
    """截取单个 cell 中心区域的彩色 patch。

    输入：
    - frame_bgr: 已对齐到协议网格的 BGR 图像。
    - row/col: cell 坐标。
    - module: 单个 cell 的像素边长。
    - sample_ratio: 中心采样窗口占 cell 边长的比例。

    输出：
    - 当前 cell 的中心彩色 patch；若中心窗口为空，则退回整格 patch。

    原理/流程：
    - 先按 `sample_ratio` 计算中心裁剪边界；
    - 优先避开 warp 边缘和相邻模块串色区域；
    - 若裁剪后窗口过小，再退回完整模块，避免返回空 patch。
    """

    margin = int(round(module * (1.0 - sample_ratio) / 2.0))
    top = row * module + margin
    left = col * module + margin
    bottom = (row + 1) * module - margin
    right = (col + 1) * module - margin
    patch = frame_bgr[top:bottom, left:right]
    if patch.size == 0:
        patch = frame_bgr[row * module : (row + 1) * module, col * module : (col + 1) * module]
    return patch


def _apply_channel_gains(colors_bgr: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """对 BGR 颜色样本应用逐通道增益校正。

    输入：
    - colors_bgr: 任意形状的 BGR 浮点数组，最后一维必须为 3。
    - gains: 长度为 3 的通道增益，顺序同样为 BGR。

    输出：
    - 应用增益并裁剪到 `[0, 255]` 后的 BGR 浮点数组。

    原理/流程：
    - 用白色参考块估计当前帧的通道偏色；
    - 再将所有采样颜色统一乘以增益，做轻量白平衡；
    - 这里只做最小必要校正，不引入复杂矩阵颜色变换。
    """

    corrected = np.asarray(colors_bgr, dtype=np.float32) * np.asarray(gains, dtype=np.float32)
    return np.clip(corrected, 0.0, 255.0)


def _bgr_to_lab(colors_bgr: np.ndarray) -> np.ndarray:
    """将 BGR 颜色样本转换到 Lab 空间。

    输入：
    - colors_bgr: 任意形状的 BGR 数组，最后一维为 3。

    输出：
    - 与输入形状对应的 Lab 浮点数组。

    原理/流程：
    - 统一用 OpenCV 的 `cv2.cvtColor` 做颜色空间转换；
    - Lab 空间比直接 RGB 欧氏距离更接近人眼颜色差异感知；
    - 这一步主要服务于 WRGB 的更稳健判色。
    """

    arr = np.asarray(colors_bgr, dtype=np.float32)
    flat = arr.reshape(-1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(flat, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return lab.reshape(arr.shape)


def _estimate_wrgb_symbol_centers(
    normalized_frame: np.ndarray,
    module: int,
    palette: dict[ColorName, tuple[int, int, int]],
    sample_ratio: float,
) -> tuple[dict[ColorName, np.ndarray], dict[int, np.ndarray], np.ndarray]:
    """估计当前帧的 WRGB 参考色中心与白平衡增益。

    输入：
    - normalized_frame: 已对齐到协议网格的彩色图。
    - module: 单个 cell 的像素边长。
    - palette: WRGB 方案的目标 palette。
    - sample_ratio: 参考块中心采样比例。

    输出：
    - `reference_centers`: 每种参考色经白平衡后的 BGR 中心。
    - `symbol_centers_lab`: 每个 WRGB symbol 对应的 Lab 颜色中心。
    - `channel_gains`: 由白色参考块估计出的 BGR 通道增益。

    原理/流程：
    - 先在 4 个 `4x4` 参考块上做中心采样并求均值；
    - 再用白色参考块与目标白色比值估计通道增益；
    - 最后把参考色中心转换到 Lab，供后续 payload 判色使用。
    """

    raw_reference_centers: dict[ColorName, np.ndarray] = {}
    for color_name in WRGB_REFERENCE_COLOR_ORDER:
        samples = [
            _sample_module_mean(normalized_frame, row, col, module, sample_ratio)
            for row, col in WRGB_REFERENCE_BLOCKS[color_name]
        ]
        raw_reference_centers[color_name] = np.mean(np.asarray(samples, dtype=np.float32), axis=0)

    observed_white = np.clip(raw_reference_centers["white"], 1.0, 255.0)
    target_white = np.asarray(_rgb_to_bgr(palette["white"]), dtype=np.float32)
    channel_gains = np.clip(target_white / observed_white, 0.6, 1.8)

    reference_centers = {
        color_name: _apply_channel_gains(center, channel_gains)
        for color_name, center in raw_reference_centers.items()
    }
    symbol_centers_lab = {
        symbol: _bgr_to_lab(reference_centers[color_name]).reshape(3)
        for symbol, color_name in WRGB_SYMBOL_TO_COLOR.items()
    }
    return reference_centers, symbol_centers_lab, channel_gains


def _classify_wrgb_patch(
    patch_bgr: np.ndarray,
    symbol_centers_lab: dict[int, np.ndarray],
    channel_gains: np.ndarray,
) -> int:
    """联合像素投票与均值距离，对单个 WRGB patch 做判色。

    输入：
    - patch_bgr: 某个 payload cell 的中心彩色 patch。
    - symbol_centers_lab: 每个 WRGB symbol 的 Lab 颜色中心。
    - channel_gains: 白色参考块估计出的通道增益。

    输出：
    - 当前 cell 对应的 WRGB symbol，取值范围为 `0..3`。

    原理/流程：
    - 先对白平衡后的 patch 做逐像素 Lab 距离计算，得到投票结果；
    - 再对 patch 均值做一次 Lab 最近邻分类；
    - 若两者一致则直接采信；若不一致，则优先使用占比更高的投票结果。
    """

    return _classify_wrgb_patch_with_metrics(
        patch_bgr,
        symbol_centers_lab,
        channel_gains,
    ).symbol


def _classify_wrgb_patch_with_metrics(
    patch_bgr: np.ndarray,
    symbol_centers_lab: dict[int, np.ndarray],
    channel_gains: np.ndarray,
) -> WRGBDecision:
    """联合像素投票与均值距离，对单个 WRGB patch 判色并输出置信度。

    输入：
    - patch_bgr: 某个 payload cell 的中心彩色 patch。
    - symbol_centers_lab: 每个 WRGB symbol 的 Lab 颜色中心。
    - channel_gains: 白色参考块估计出的通道增益。

    输出：
    - `WRGBDecision`，包含类别、投票占比和综合置信度。

    原理/流程：
    - 先对白平衡后的 patch 做逐像素 Lab 距离投票；
    - 再对 patch 均值做一次 Lab 最近邻分类；
    - 若两者一致，则显著提高置信度；
    - 若两者不一致，则根据投票占比与均值裕量共同决定置信度。
    """

    corrected_patch = _apply_channel_gains(patch_bgr, channel_gains).astype(np.uint8)
    pixel_lab = _bgr_to_lab(corrected_patch.reshape(-1, 3)).reshape(-1, 3)
    symbols = tuple(sorted(symbol_centers_lab))
    center_matrix = np.stack([symbol_centers_lab[symbol] for symbol in symbols], axis=0)

    pixel_distances = np.linalg.norm(pixel_lab[:, None, :] - center_matrix[None, :, :], axis=2)
    pixel_labels = pixel_distances.argmin(axis=1)
    vote_counts = np.bincount(pixel_labels, minlength=len(symbols)).astype(np.float32)
    vote_order = np.argsort(vote_counts)[::-1]
    vote_winner_index = int(vote_order[0])
    vote_runner_index = int(vote_order[1]) if len(vote_order) > 1 else vote_winner_index
    vote_ratio = float(vote_counts[vote_winner_index] / max(1.0, float(pixel_labels.size)))
    vote_margin = float(vote_counts[vote_winner_index] - vote_counts[vote_runner_index]) / max(
        1.0,
        float(pixel_labels.size),
    )
    vote_symbol = int(symbols[vote_winner_index])

    mean_lab = _bgr_to_lab(corrected_patch.reshape(-1, 3).mean(axis=0)).reshape(3)
    mean_distances = np.linalg.norm(center_matrix - mean_lab[None, :], axis=1)
    mean_order = np.argsort(mean_distances)
    mean_best = float(mean_distances[int(mean_order[0])])
    mean_second = float(mean_distances[int(mean_order[1])]) if len(mean_order) > 1 else mean_best
    mean_margin = (mean_second - mean_best) / max(1.0, mean_second)
    mean_symbol = int(symbols[int(mean_order[0])])

    if vote_symbol == mean_symbol:
        symbol = vote_symbol
        confidence = min(1.0, 0.55 * vote_ratio + 0.20 * vote_margin + 0.25 * max(0.0, mean_margin))
    elif vote_ratio >= 0.60:
        symbol = vote_symbol
        confidence = min(0.85, 0.60 * vote_ratio + 0.15 * vote_margin + 0.10 * max(0.0, mean_margin))
    else:
        symbol = mean_symbol
        confidence = min(0.80, 0.35 * vote_ratio + 0.10 * vote_margin + 0.35 * max(0.0, mean_margin))

    return WRGBDecision(
        symbol=int(symbol),
        confidence=float(max(0.0, confidence)),
        vote_ratio=float(max(0.0, min(1.0, vote_ratio))),
        mean_margin=float(max(0.0, min(1.0, mean_margin))),
    )


def _build_wrgb_symbol_maps(
    required_symbols: int,
    decisions: list[WRGBDecision],
) -> tuple[np.ndarray, np.ndarray]:
    """构建 WRGB 符号图与置信度图。

    输入：
    - required_symbols: 当前帧实际需要恢复的 payload symbol 数量。
    - decisions: 对应 payload cell 的逐单元判色结果。

    输出：
    - `symbol_map`: `GRID_SIZE x GRID_SIZE` 的整型符号图，未使用位置为 `-1`
    - `confidence_map`: 与 `symbol_map` 对应的浮点置信度图

    原理/流程：
    - payload 区按当前判色结果填入；
    - WRGB 顶部参考块直接写入理论类别，作为硬锚点；
    - 参考块置信度固定为 1，后续邻域修正只会借助它们，不会改它们。
    """

    symbol_map = np.full((GRID_SIZE, GRID_SIZE), -1, dtype=np.int16)
    confidence_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    for color_name in WRGB_REFERENCE_COLOR_ORDER:
        symbol = WRGB_COLOR_TO_SYMBOL[color_name]
        for row, col in WRGB_REFERENCE_BLOCKS[color_name]:
            symbol_map[row, col] = symbol
            confidence_map[row, col] = 1.0

    for index, (row, col) in enumerate(WRGB_PAYLOAD_CELLS[:required_symbols]):
        decision = decisions[index]
        symbol_map[row, col] = int(decision.symbol)
        confidence_map[row, col] = float(decision.confidence)

    return symbol_map, confidence_map


def _apply_wrgb_low_confidence_correction(
    symbol_map: np.ndarray,
    confidence_map: np.ndarray,
    *,
    required_symbols: int,
    low_conf_threshold: float = 0.52,
    high_conf_threshold: float = 0.82,
    min_support_count: int = 3,
) -> np.ndarray:
    """对低置信 WRGB payload 单元做保守的邻域修正。

    输入：
    - symbol_map: 当前帧的整图符号分类结果
    - confidence_map: 与 `symbol_map` 对应的置信度
    - required_symbols: 当前帧真实参与解码的 payload symbol 数量
    - low_conf_threshold: 低置信阈值，低于该值才考虑修正
    - high_conf_threshold: 高置信阈值，邻域支持必须高于该值才可作为可信种子
    - min_support_count: 同类高置信邻居最少数量

    输出：
    - 邻域修正后的 payload symbol 数组

    原理/流程：
    - 仅对低置信 payload 单元进行处理；
    - 只参考 8 邻域中的高置信单元与参考块；
    - 必须出现明显多数支持才改写，避免对随机 payload 做过强平滑。
    """

    corrected_map = symbol_map.copy()
    payload_cells = WRGB_PAYLOAD_CELLS[:required_symbols]

    for row, col in payload_cells:
        current_symbol = int(corrected_map[row, col])
        current_conf = float(confidence_map[row, col])
        if current_symbol < 0 or current_conf >= low_conf_threshold:
            continue

        neighbor_support: dict[int, float] = {}
        neighbor_counts: dict[int, int] = {}
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = row + dr
                cc = col + dc
                if not (0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE):
                    continue
                neighbor_symbol = int(corrected_map[rr, cc])
                neighbor_conf = float(confidence_map[rr, cc])
                if neighbor_symbol < 0 or neighbor_conf < high_conf_threshold:
                    continue
                neighbor_support[neighbor_symbol] = neighbor_support.get(neighbor_symbol, 0.0) + neighbor_conf
                neighbor_counts[neighbor_symbol] = neighbor_counts.get(neighbor_symbol, 0) + 1

        if not neighbor_support:
            continue

        candidate_symbol, candidate_weight = max(
            neighbor_support.items(),
            key=lambda item: (item[1], neighbor_counts[item[0]], -item[0]),
        )
        candidate_count = neighbor_counts[candidate_symbol]
        total_weight = float(sum(neighbor_support.values()))
        candidate_ratio = candidate_weight / max(1e-6, total_weight)

        if (
            candidate_symbol != current_symbol
            and candidate_count >= min_support_count
            and candidate_ratio >= 0.72
        ):
            corrected_map[row, col] = int(candidate_symbol)

    return np.array(
        [int(corrected_map[row, col]) for row, col in payload_cells],
        dtype=np.uint8,
    )


def _frame_to_grid_bgr(frame_bgr: np.ndarray) -> tuple[np.ndarray, int]:
    """将输入图像规范化为协议网格倍数尺寸的彩色图。

    输入：
    - frame_bgr: 原始彩色帧，可以已经是 `GRID_SIZE` 倍数尺寸，也可以是网格尺寸图。

    输出：
    - 规范化后的 BGR 图像
    - 对应的单个 cell 像素边长

    原理/流程：
    - 如果输入已经是 `GRID_SIZE x GRID_SIZE`，则视作每个 cell 只有 1 个像素。
    - 如果输入是整数倍尺寸，则保留原尺寸用于中心窗口采样。
    - 否则先拉回整数网格尺寸，保证后续按 cell 采样时不会错位。
    """

    arr = np.asarray(frame_bgr, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("颜色解码输入必须是 BGR 三通道图像")

    if arr.shape[:2] == (GRID_SIZE, GRID_SIZE):
        return arr, 1

    if arr.shape[0] % GRID_SIZE == 0 and arr.shape[1] % GRID_SIZE == 0:
        module = min(arr.shape[0] // GRID_SIZE, arr.shape[1] // GRID_SIZE)
        size = module * GRID_SIZE
        if arr.shape[0] != size or arr.shape[1] != size:
            arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_NEAREST)
        return arr, module

    resized = cv2.resize(arr, (GRID_SIZE * 10, GRID_SIZE * 10), interpolation=cv2.INTER_NEAREST)
    return resized, 10


def decode_color_bits(
    frame_bgr: np.ndarray,
    bit_length: int,
    *,
    layout: ColorCodecLayout | None = None,
    palette: dict[ColorName, tuple[int, int, int]] | None = None,
    sample_ratio: float = 0.5,
) -> np.ndarray:
    """从理想对齐的彩色协议帧中恢复 payload bit 流。

    输入：
    - frame_bgr: 已经和协议网格对齐的 BGR 彩色帧。
    - bit_length: 期望恢复的 payload bit 数。
    - layout: 颜色布局配置。
    - palette: 颜色表；为空时使用方案 A。
    - sample_ratio: 中心采样窗口占 cell 边长的比例。

    输出：
    - 恢复出的 payload bit 数组。

    原理/流程：
    - 先在参考色块位置采样，估计每帧实际颜色中心。
    - 再对每个 payload cell 采样中心颜色。
    - 使用最近颜色中心分类，将颜色 symbol 还原成 2 bit。
    """

    actual_layout = layout or ColorCodecLayout()
    actual_palette = _normalize_palette(palette)
    normalized_frame, module = _frame_to_grid_bgr(frame_bgr)

    ref_cells = get_reference_cells(actual_layout)
    payload_cells = get_color_payload_cells(actual_layout)
    repeats = actual_layout.ref_repeats

    reference_centers: dict[ColorName, np.ndarray] = {}
    for color_index, color_name in enumerate(REFERENCE_COLOR_ORDER):
        start = color_index * repeats
        end = start + repeats
        samples = [
            _sample_module_mean(normalized_frame, row, col, module, sample_ratio)
            for row, col in ref_cells[start:end]
        ]
        reference_centers[color_name] = np.mean(np.asarray(samples, dtype=np.float32), axis=0)

    symbol_centers = {
        symbol: reference_centers[color_name]
        for symbol, color_name in SYMBOL_TO_COLOR.items()
    }

    required_symbols = (bit_length + 1) // 2
    decoded_symbols: list[int] = []
    for row, col in payload_cells[:required_symbols]:
        sample = _sample_module_mean(normalized_frame, row, col, module, sample_ratio)
        best_symbol = min(
            symbol_centers,
            key=lambda symbol: float(np.linalg.norm(sample - symbol_centers[symbol])),
        )
        decoded_symbols.append(int(best_symbol))

    return _unpair_symbols(decoded_symbols, bit_length)


__all__ = [
    "COLOR_TO_SYMBOL",
    "COLOR_ACTUAL_ECC_BYTES",
    "COLOR_DATA_SIZE_LIMIT",
    "COLOR_MAX_DATA_BYTES",
    "COLOR_TOTAL_CAPACITY_BITS",
    "ColorCodecLayout",
    "ColorDecodeResult",
    "PALETTE_A_RGB",
    "PALETTE_B_RGB",
    "PALETTE_WRGB_RGB",
    "REFERENCE_COLOR_ORDER",
    "SYMBOL_TO_COLOR",
    "WRGB_ACTUAL_ECC_BYTES",
    "WRGB_COLOR_TO_SYMBOL",
    "WRGB_DATA_SIZE_LIMIT",
    "WRGB_MAX_DATA_BYTES",
    "WRGB_REFERENCE_COLOR_ORDER",
    "WRGB_SYMBOL_TO_COLOR",
    "WRGB_TOTAL_CAPACITY_BITS",
    "decode_color_frame",
    "decode_color_bits",
    "decode_wrgb_frame",
    "decode_wrgb_bits",
    "encode_color_bin",
    "encode_color_bitstream",
    "encode_color_payload_frame",
    "encode_wrgb_bitstream",
    "encode_wrgb_payload_frame",
    "estimate_color_payload_capacity_bytes",
    "get_color_payload_capacity_bits",
    "get_color_payload_cells",
    "get_reference_cells",
    "get_wrgb_payload_capacity_bits",
    "get_wrgb_payload_cells",
    "get_wrgb_reference_cells",
    "render_color_protocol_frames_from_bits",
    "render_color_frame",
    "render_color_frames_from_bits",
    "render_wrgb_frame",
    "save_color_frame",
    "save_color_protocol_video",
    "save_color_video",
    "save_wrgb_protocol_video",
]
