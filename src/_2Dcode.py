"""
二维码协议编解码文件
方法：
  - encode_bin(path) 传入路径，返回一个矩阵列表
  - decode_image(image) 传入二维码，返回数据与合法性标志（均为二进制）
"""

from __future__ import annotations

import numpy as np

from utils.showimg import *
from pathlib import Path
from config import *
import cv2, binascii

def bytes_to_bits(data: bytes) -> np.ndarray:
    """将字节数据转换为bit01列表"""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

def _build_big_finder() -> np.ndarray:
    finder = np.zeros((BIG_FINDER_SIZE, BIG_FINDER_SIZE), dtype=np.uint8)
    for r in range(BIG_FINDER_SIZE):
        finder[r, 0] = finder[r, -1] = finder[r, 1] = finder[r, -2] = 1
    for c in range(BIG_FINDER_SIZE):
        finder[0, c] = finder[-1, c] = finder[1, c] = finder[-2, c] = 1
    finder[4:10, 4:10] = 1
    return finder


def _build_small_finder() -> np.ndarray:
    finder = np.zeros((SMALL_FINDER_SIZE, SMALL_FINDER_SIZE), dtype=np.uint8)
    finder[0, :] = 1
    finder[-1, :] = 1
    finder[:, 0] = 1
    finder[:, -1] = 1
    finder[2:6, 2:6] = 1
    return finder


BIG_FINDER = _build_big_finder()
SMALL_FINDER = _build_small_finder()

def _paste(dst: np.ndarray, src: np.ndarray, top: int, left: int) -> None:
    h, w = src.shape
    dst[top : top + h, left : left + w] = src


def make_base_grid() -> np.ndarray:
    """创建基本矩阵"""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

    # 放置定位格
    _paste(grid, BIG_FINDER, QUIET_WIDTH, QUIET_WIDTH)
    _paste(grid, BIG_FINDER, QUIET_WIDTH, GRID_SIZE - QUIET_WIDTH - BIG_FINDER_SIZE)
    _paste(grid, BIG_FINDER, GRID_SIZE - QUIET_WIDTH - BIG_FINDER_SIZE, QUIET_WIDTH)
    _paste(
        grid,
        SMALL_FINDER,
        GRID_SIZE - QUIET_WIDTH - SMALL_FINDER_SIZE,
        GRID_SIZE - QUIET_WIDTH - SMALL_FINDER_SIZE,
    )
    return grid

BASE_GRID = make_base_grid()

def _iter_cells(bounds: tuple[int, int, int, int]):
    rs, re, cs, ce = bounds
    for r in range(rs, re):
        if r % 2 == 1:
            _iter = range(ce-1, cs-1, -1)
        else:
            _iter = range(cs, ce)
        for c in _iter:
            yield r, c

def draw(vis, x1, y1, x2, y2):
    for r in range(x1, x2 + 1):
        for c in range(y1, y2 + 1):
            vis[r, c] = 1

def _iter_data_cells():
    hrs, hre, hcs, hce = SMALL_FINDER_BOUNDS
    vis = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    draw(vis, 18, 2, 89, 18)
    draw(vis, 12, 18, 18, 28)
    draw(vis, 2, 28, 18, 89)
    draw(vis, 18, 18, 105, 105)
    
    for r in range(hrs-1, hre):
        for c in range(hcs-1, hce):
            vis[r, c] = 0
        
    for r in range(GRID_SIZE):
        if r % 2 == 1:
            _iter = range(GRID_SIZE-1, -1, -1)
        else:
            _iter = range(GRID_SIZE)
        for c in _iter:
            if vis[r][c]:
                yield r, c
    
    # for r in range(GRID_SIZE):
    #     for c in range(GRID_SIZE):
    #         if vis[r][c]:
    #             yield r, c


DATA_ITER = list(_iter_data_cells())
INFO_ITER = list(_iter_cells(HEADER_BOUNDS))

def _fill_region_random(grid: np.ndarray, cells, rng: np.random.Generator) -> None:
    for r, c in cells:
        grid[r, c] = np.uint8(rng.integers(0, 2))


def generate_random_compliant_qr(
    seed: int | None = None,
    fill_header: bool = True,
    fill_check: bool = True,
    fill_data: bool = True,
) -> np.ndarray:
    """
    生成一个随机二维码
    Args:
        seed: 随机种子
        fill_header: 是否放置信息头
        fill_check: 是否放置校验区
        fill_data: 是否放置数据区
    """
    grid = make_base_grid()
    rng = np.random.default_rng(seed)

    if fill_header:
        _fill_region_random(grid, _iter_cells(HEADER_BOUNDS), rng)
    if fill_check:
        _fill_region_random(grid, _iter_cells(CHECK_BOUNDS), rng)
    if fill_data:
        _fill_region_random(grid, _iter_data_cells_excluding_small_finder(), rng)

    return grid


def preview_random_compliant_qr(
    seed: int | None = None,
    pixel_per_cell: int = 8,
    window_name: str = "Random Compliant QR",
) -> np.ndarray:
    """Generate and display one random compliant QR-like matrix."""
    grid = generate_random_compliant_qr(seed=seed)
    show_binary_matrix(grid, pixel_per_cell=pixel_per_cell, window_name=window_name)
    return grid

def save_test_frames(grids: np.ndarray, out_dir: str = "outputs/test_frames") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i, grid in enumerate(grids):
        # 1) 保存原始矩阵（0/1）
        np.save(out / f"frame_{i:04d}.npy", grid)

        # 2) 保存可视化图片（1黑0白，放大10倍）
        img = matrix_to_bw_image(grid, pixel_per_cell=10)
        ok = cv2.imwrite(str(out / f"frame_{i:04d}.png"), img)
        if not ok:
            raise RuntimeError(f"保存失败: frame_{i:04d}.png")

def get_infoheader_from_bits(len: int, index: int) -> np.ndarray:
    """由bit流即序列号生成信息头矩阵"""
    # 信息头包括：
    # 1. 帧序列号
    # 2. 有效数据长度

    len_bits = len.to_bytes(2, byteorder="big")
    index_bits = index.to_bytes(2, byteorder="big")
    return bytes_to_bits(len_bits + index_bits)
    

def get_checkcode_from_bits(data : bytes) -> np.ndarray:
    """由bit流生成校验码"""
    # 校验区包括：CRC32校验码
    return bytes_to_bits(binascii.crc32(data).to_bytes(4, byteorder="big"))

def get_from_bits(bits: np.ndarray, index : int) -> np.ndarray:
    """从bits中提取出二维码矩阵"""
    grid = make_base_grid()
    for idx, (r, c) in enumerate(DATA_ITER):
        if idx >= len(bits):
            break
        grid[r, c] = bits[idx]
    
    check = get_checkcode_from_bits(np.packbits(bits))
    info = get_infoheader_from_bits(len(bits), index)
    header = np.concatenate((check, info))
    
    for idx, (r, c) in enumerate(INFO_ITER):
        if idx >= len(header):
            break
        grid[r, c] = header[idx]
    
    return grid

def encode_bin(path) -> list[np.ndarray]:
    """
    将二进制文件编码为二维码矩阵列表\n
    Args:
        path: 输入二进制文件路径，长度 ≤ 10MB
    """
    with open(path, "rb") as f:
        data = f.read()
    
    bit_data = bytes_to_bits(data)
    data_ls = [np.array(bit_data[i:i+DATA_SIZE_LIMIT], dtype=np.uint8) for i in range(0, len(bit_data), DATA_SIZE_LIMIT)]
    
    grids = np.array([get_from_bits(bits, idx) for idx, bits in enumerate(data_ls)])
    
    return grids
    
    
def decode_image(img: np.ndarray) -> tuple[bytes, bytes]:
    """
    将二维码矩阵解码为二进制数据与合法性标志\n
    将返回数据与合法性标志的元组，均为二进制数据
    Args:
        img: 二维二维码矩阵
    """

if __name__ == "__main__":
    tmp = encode_bin("pushtest.txt")
    print(len(tmp))
    save_test_frames(tmp)
    

