"""
"""


from __future__ import annotations

import numpy as np

from utils.showimg import show_binary_matrix


GRID_SIZE = 108
QUIET_WIDTH = 2

BIG_FINDER_SIZE = 14
SMALL_FINDER_SIZE = 8

# Region bounds: (row_start, row_end_exclusive, col_start, col_end_exclusive)
HEADER_BOUNDS = (2, 16, 18, 90)   # 14 x 72
CHECK_BOUNDS = (18, 90, 2, 16)    # 72 x 14
DATA_BOUNDS = (18, 106, 18, 106)  # 88 x 88
SMALL_FINDER_BOUNDS = (98, 106, 98, 106)  # 8 x 8, inside data area


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


def _iter_cells(bounds: tuple[int, int, int, int]):
    rs, re, cs, ce = bounds
    for r in range(rs, re):
        for c in range(cs, ce):
            yield r, c


def _iter_data_cells_excluding_small_finder():
    rs, re, cs, ce = DATA_BOUNDS
    frs, fre, fcs, fce = SMALL_FINDER_BOUNDS
    for r in range(rs, re):
        for c in range(cs, ce):
            if frs-1 <= r < fre and fcs-1 <= c < fce:
                continue
            yield r, c


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


if __name__ == "__main__":
    preview_random_compliant_qr(seed=None, pixel_per_cell=8)

