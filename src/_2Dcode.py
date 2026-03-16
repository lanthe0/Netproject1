"""
二维码协议编解码文件
方法：
  - encode_bin(path) 传入路径，返回一个矩阵列表
  - decode_image(image，out_bin_path, vout_bin_path) 传入二维码和路径，写入数据与合法性标志（均为二进制）
"""

from __future__ import annotations

from pathlib import Path

import binascii
import cv2
import numpy as np
import reedsolo

from config import *
from utils.showimg import *


def bytes_to_bits(data: bytes) -> np.ndarray:
    """将字节数据转换为 bit01 列表，兼容多种输入类型。"""
    if not isinstance(data, (bytes, bytearray)):
        data = bytes(data)
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
    finder[2:5, 2:5] = 1
    return finder


BIG_FINDER = _build_big_finder()
SMALL_FINDER = _build_small_finder()


def _paste(dst: np.ndarray, src: np.ndarray, top: int, left: int) -> None:
    h, w = src.shape
    dst[top : top + h, left : left + w] = src


def make_base_grid() -> np.ndarray:
    """创建基本矩阵。"""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

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
            indices = range(ce - 1, cs - 1, -1)
        else:
            indices = range(cs, ce)
        for c in indices:
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

    for r in range(hrs - 1, hre):
        for c in range(hcs - 1, hce):
            vis[r, c] = 0

    for r in range(GRID_SIZE):
        if r % 2 == 1:
            indices = range(GRID_SIZE - 1, -1, -1)
        else:
            indices = range(GRID_SIZE)
        for c in indices:
            if vis[r][c]:
                yield r, c


DATA_ITER = list(_iter_data_cells())
INFO_ITER = list(_iter_cells(HEADER_BOUNDS))
if len(DATA_ITER) != DATA_SIZE_LIMIT:
    raise ValueError(f"数据区容量 {len(DATA_ITER)} 不等于预设值 {DATA_SIZE_LIMIT}")


def save_test_frames(grids: np.ndarray, out_dir: str = "output/test_frames") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i, grid in enumerate(grids):
        np.save(out / f"frame_{i:04d}.npy", grid)
        img = matrix_to_bw_image(grid, pixel_per_cell=10)
        ok = cv2.imwrite(str(out / f"frame_{i:04d}.png"), img)
        if not ok:
            raise RuntimeError(f"保存失败: frame_{i:04d}.png")


def get_infoheader_from_bits(length: int, index: int) -> np.ndarray:
    """由 bit 流和序列号生成信息头矩阵。"""
    len_bits = length.to_bytes(2, byteorder="big")
    index_bits = index.to_bytes(2, byteorder="big")
    return bytes_to_bits(len_bits + index_bits)


def get_checkcode_from_bits(data: bytes) -> np.ndarray:
    """由 bit 流生成 CRC32 校验码。"""
    return bytes_to_bits(binascii.crc32(data).to_bytes(4, byteorder="big"))


def get_from_bits(bits: np.ndarray, index: int) -> np.ndarray:
    """从 bits 中提取出二维码矩阵。"""
    rs = reedsolo.RSCodec(ECC_BYTES)
    grid = BASE_GRID.copy()

    raw_bytes = np.packbits(bits).tobytes()
    ecc_payload_bytes = rs.encode(raw_bytes)
    if not isinstance(ecc_payload_bytes, bytes):
        ecc_payload_bytes = bytes(ecc_payload_bytes)
    ecc_bits = bytes_to_bits(ecc_payload_bytes)

    if len(ecc_bits) > len(DATA_ITER):
        raise ValueError(
            f"加上纠错码后数据长度 {len(ecc_bits)} 超过二维码容量 {len(DATA_ITER)}"
        )

    for idx, (r, c) in enumerate(DATA_ITER):
        if idx >= len(ecc_bits):
            break
        grid[r, c] = ecc_bits[idx]

    check = get_checkcode_from_bits(raw_bytes)
    info = get_infoheader_from_bits(len(bits), index)
    header = np.concatenate((check, info))

    for idx, (r, c) in enumerate(INFO_ITER):
        if idx >= len(header):
            break
        grid[r, c] = header[idx]

    return grid


def encode_bin(path) -> np.ndarray:
    """
    将二进制文件编码为二维码矩阵列表。

    Args:
        path: 输入二进制文件路径，长度 <= 10MB
    """
    with open(path, "rb") as f:
        data = f.read()

    bit_data = bytes_to_bits(data)
    data_ls = [
        np.array(bit_data[i : i + DATA_SIZE_LIMIT], dtype=np.uint8)
        for i in range(0, len(bit_data), DATA_SIZE_LIMIT)
    ]
    return np.array([get_from_bits(bits, idx) for idx, bits in enumerate(data_ls)])


def get_total_length_from_grids(grids: np.ndarray) -> int:
    """通过解析信息头估计原始文件总字节数。"""
    total_bits = 0

    for grid in grids:
        header_bits = np.array(
            [grid[r, c] for (r, c) in INFO_ITER[:HEADER_SIZE]], dtype=np.uint8
        )
        info_bits = header_bits[32:64]
        bit_len = 0
        for b in info_bits[:16]:
            bit_len = (bit_len << 1) | int(b)
        total_bits += bit_len

    return total_bits // 8


def decode_image(imgs: np.ndarray, out_bin_path: str, out_vbin_path: str) -> None:
    """
    将二维码矩阵解码为二进制数据与合法性标志。

    Args:
        imgs: 二维二维码矩阵序列
    """
    rs = reedsolo.RSCodec(ECC_BYTES)

    def _to_grid(frame) -> np.ndarray:
        """将输入帧转换为 108x108 二值矩阵（1 黑 0 白）。"""
        if isinstance(frame, np.ndarray) and frame.dtype == object:
            try:
                frame = frame.astype(np.uint8)
            except (TypeError, ValueError):
                return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

        if arr.shape == (GRID_SIZE, GRID_SIZE):
            if arr.max() <= 1:
                return (arr > 0).astype(np.uint8)
            return (arr < 128).astype(np.uint8)

        if arr.ndim != 2:
            return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

        if arr.shape[0] % GRID_SIZE != 0 or arr.shape[1] % GRID_SIZE != 0:
            arr = cv2.resize(arr, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_NEAREST)
            return (arr < 128).astype(np.uint8)

        module = arr.shape[0] // GRID_SIZE
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                block = arr[r * module : (r + 1) * module, c * module : (c + 1) * module]
                grid[r, c] = 1 if np.mean(block) < 128 else 0
        return grid

    def _bits_to_int(bits: np.ndarray) -> int:
        value = 0
        for bit in bits.astype(np.uint8):
            value = (value << 1) | int(bit)
        return value

    def _pack_bits(bits: list[int]) -> bytes:
        if not bits:
            return b""
        arr = np.array(bits, dtype=np.uint8)
        if arr.size % 8 != 0:
            arr = np.pad(arr, (0, 8 - arr.size % 8), mode="constant", constant_values=0)
        return np.packbits(arr).tobytes()

    arr_imgs = np.asarray(imgs, dtype=object)
    if arr_imgs.ndim == 2:
        frames = [arr_imgs]
    else:
        frames = [arr_imgs[i] for i in range(len(arr_imgs))]

    decoded_frames: dict[int, tuple[np.ndarray, bool]] = {}
    truncated_len_frames = 0

    for frame in frames:
        grid = _to_grid(frame)

        header_bits = np.array(
            [grid[r, c] for (r, c) in INFO_ITER[:HEADER_SIZE]], dtype=np.uint8
        )
        if header_bits.size < HEADER_SIZE:
            continue

        crc_bits = header_bits[:32]
        info_bits = header_bits[32:64]
        expected_crc = _bits_to_int(crc_bits)
        bit_len = _bits_to_int(info_bits[:16])
        frame_idx = _bits_to_int(info_bits[16:32])

        if bit_len > len(DATA_ITER):
            bit_len = len(DATA_ITER)
            truncated_len_frames += 1

        byte_len = (bit_len + 7) // 8
        total_byte_len = byte_len + ECC_BYTES
        total_bit_len = total_byte_len * 8
        ecc_payload_bits = np.array(
            [grid[r, c] for (r, c) in DATA_ITER[:total_bit_len]], dtype=np.uint8
        )

        valid = False
        payload_bits = np.zeros(bit_len, dtype=np.uint8)

        try:
            ecc_bytes = np.packbits(ecc_payload_bits).tobytes()
            corrected_raw_bytes, _, _ = rs.decode(ecc_bytes)
            corrected_bytes = bytes(corrected_raw_bytes)
            payload_bits = bytes_to_bits(corrected_bytes)[:bit_len]
            actual_crc = binascii.crc32(corrected_bytes[:byte_len]) & 0xFFFFFFFF
            valid = actual_crc == expected_crc
        except reedsolo.ReedSolomonError:
            pass

        old = decoded_frames.get(frame_idx)
        if old is None or (not old[1] and valid):
            decoded_frames[frame_idx] = (payload_bits, valid)

    all_data_bits: list[int] = []
    all_valid_bits: list[int] = []

    for idx in sorted(decoded_frames.keys()):
        bits, valid = decoded_frames[idx]
        bits_list = bits.astype(np.uint8).tolist()
        all_data_bits.extend(bits_list)
        all_valid_bits.extend(([1] * len(bits_list)) if valid else ([0] * len(bits_list)))

    if truncated_len_frames > 0:
        print(
            "[decode] warning: "
            f"{truncated_len_frames} frame(s) had declared length beyond DATA_ITER "
            "capacity; truncated during decode"
        )

    with open(out_bin_path, "wb") as f_out:
        f_out.write(_pack_bits(all_data_bits))

    with open(out_vbin_path, "wb") as f_vout:
        f_vout.write(_pack_bits(all_valid_bits))


def compare_files(file1_path, file2_path, chunk_size=8192):
    """二进制模式比较两个文件是否完全相同。"""
    with open(file1_path, "rb") as f1, open(file2_path, "rb") as f2:
        while True:
            b1 = f1.read(chunk_size)
            b2 = f2.read(chunk_size)
            if b1 != b2:
                return False
            if not b1:
                return True


def verify_saved_frames(
    frames_dir: str,
    original_file_path: str,
    decoded_file_path: str,
    validity_file_path: str,
) -> bool:
    """测试函数：读取已保存二维码图片，解码并与原始文件做字节级对比。"""
    frame_paths = sorted(Path(frames_dir).glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"未找到二维码图片: {frames_dir}")

    frames = []
    for p in frame_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"读取图片失败: {p}")
        frames.append(img)

    decode_image(np.asarray(frames, dtype=object), decoded_file_path, validity_file_path)

    with open(original_file_path, "rb") as f:
        original = f.read()
    with open(decoded_file_path, "rb") as f:
        decoded = f.read()

    same = original == decoded
    print(f"[verify] frame count: {len(frame_paths)}")
    print(f"[verify] original bytes: {len(original)}")
    print(f"[verify] decoded  bytes: {len(decoded)}")
    print(f"[verify] equal: {same}")

    if not same:
        mismatch = next(
            (i for i, (a, b) in enumerate(zip(original, decoded)) if a != b),
            None,
        )
        if mismatch is None and len(original) != len(decoded):
            mismatch = min(len(original), len(decoded))
        print(f"[verify] first mismatch index: {mismatch}")
    return same


if __name__ == "__main__":
    tmp = encode_bin("../data/test_pattern.bin")
    print(len(tmp))
    save_test_frames(tmp)
    verify_saved_frames(
        frames_dir="output/test_frames",
        original_file_path="data/test_pattern.bin",
        decoded_file_path="output/decoded.bin",
        validity_file_path="output/vout.bin",
    )
