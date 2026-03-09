import cv2
import numpy as np
from config import *

def bin2img(data: bytes):
    """
    将二进制数据转换为图片组
    严格遵循 config.py 中的 HEADER_BOUNDS, CHECK_BOUNDS, DATA_BOUNDS\n
    Args:
        data: 二进制数据
    """
    # 1. 计算单帧容量 (比特)
    # 数据区 88x88，排除右下角 8x8 的小定位格
    data_area_bits = (DATA_BOUNDS[1] - DATA_BOUNDS[0]) * (DATA_BOUNDS[3] - DATA_BOUNDS[2])
    small_finder_bits = (SMALL_FINDER_BOUNDS[1] - SMALL_FINDER_BOUNDS[0]) * (SMALL_FINDER_BOUNDS[3] - SMALL_FINDER_BOUNDS[2])
    usable_data_bits = data_area_bits - small_finder_bits
    
    bytes_per_frame = usable_data_bits // 8
    total_frames = (len(data) + bytes_per_frame - 1) // bytes_per_frame
    
    img_group = []
    
    for frame_idx in range(total_frames):
        # 初始化 108x108 矩阵 (默认白色 255)
        matrix = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.uint8) * 255
        
        # --- 绘制固定定位格 (二维码 V2 设计) ---
        # 左上、右上、左下大定位格 (14x14)
        matrix[2:16, 2:16] = 0
        matrix[2:16, 92:106] = 0
        matrix[92:106, 2:16] = 0
        # 右下小定位格 (8x8)
        r_s, r_e, c_s, c_e = SMALL_FINDER_BOUNDS
        matrix[r_s:r_e, c_s:c_e] = 0

        # --- 填充 Header (14x72) ---
        # 协议：[16b Index | 16b Total | 8b Len | Padding]
        chunk_start = frame_idx * bytes_per_frame
        chunk_end = min(chunk_start + bytes_per_frame, len(data))
        current_chunk = data[chunk_start:chunk_end]
        
        header_bits = format(frame_idx, '016b') + format(total_frames, '016b') + format(len(current_chunk), '08b')
        header_bits = header_bits.ljust(1008, '0') # 14*72=1008
        
        h_rs, h_re, h_cs, h_ce = HEADER_BOUNDS
        h_idx = 0
        for r in range(h_rs, h_re):
            for c in range(h_cs, h_ce):
                matrix[r, c] = 0 if header_bits[h_idx] == '1' else 255
                h_idx += 1

        # --- 填充 Data (88x88 排除小定位格) ---
        data_bits = ''.join(format(b, '08b') for b in current_chunk)
        data_bits = data_bits.ljust(usable_data_bits, '0')
        
        d_rs, d_re, d_cs, d_ce = DATA_BOUNDS
        s_rs, s_re, s_cs, s_ce = SMALL_FINDER_BOUNDS
        
        bit_ptr = 0
        for r in range(d_rs, d_re):
            for c in range(d_cs, d_ce):
                # 避开小定位格
                if not (s_rs <= r < s_re and s_cs <= c < s_ce):
                    if bit_ptr < len(data_bits):
                        matrix[r, c] = 0 if data_bits[bit_ptr] == '1' else 255
                        bit_ptr += 1
        
        img_group.append(matrix)
        
    return img_group