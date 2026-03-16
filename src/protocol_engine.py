# -*- coding: utf-8 -*-
import numpy as np
import reedsolo
from config import GRID_SIZE, HEADER_BOUNDS, DATA_BOUNDS, SMALL_FINDER_BOUNDS

class FrameEncoder:
    def __init__(self, ecc_bytes=12):
        self.grid_size = GRID_SIZE
        self.ecc_bytes = ecc_bytes
        self.rs = reedsolo.RSCodec(ecc_bytes)

    def _get_data_area_mask(self):
        """
        根据 config.py 的 DATA_BOUNDS 创建掩码，排除掉右下角的 SMALL_FINDER
        """
        mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        r_s, r_e, c_s, c_e = DATA_BOUNDS
        mask[r_s:r_e, c_s:c_e] = True
        
        # 排除掉右下角的小定位格
        sr_s, sr_e, sc_s, sc_e = SMALL_FINDER_BOUNDS
        mask[sr_s:sr_e, sc_s:sc_e] = False
        return mask

    def calculate_capacity_bits(self):
        """计算纯数据区(DATA_BOUNDS)能放多少比特"""
        mask = self._get_data_area_mask()
        return np.sum(mask)

    def encode_frame(self, frame_idx, total_frames, data_chunk):
        """
        核心逻辑：按照 config.py 的分区填充矩阵
        """
        matrix = np.ones((self.grid_size, self.grid_size), dtype=np.uint8) * 255 # 默认白色
        
        # 1. 处理 Header (14x72 = 1008 bits)
        # 协议：[16bit Index | 16bit Total | 8bit Len | Remaining Padding]
        header_bits = format(frame_idx, '016b') + format(total_frames, '016b') + format(len(data_chunk), '08b')
        header_bits = header_bits.ljust(1008, '0')
        self._fill_area(matrix, HEADER_BOUNDS, header_bits)

        # 2. 处理 Data + ECC
        # 将原始数据转为 bits 并通过 RS 编码（如果需要整体纠错）
        # 这里演示简单填充，卓越班同学建议在 DATA 结束后紧跟 ECC
        data_bits = ''.join(format(b, '08b') for b in data_chunk)
        self._fill_data_area(matrix, data_bits)

        # 3. 绘制定位格 (交给庄嘉豪的 _2Dcode 处理视觉，这里只负责数据留白或填黑)
        # 建议这里只负责返回 0/1 矩阵
        return matrix

    def _fill_area(self, matrix, bounds, bits):
        r_s, r_e, c_s, c_e = bounds
        idx = 0
        for r in range(r_s, r_e):
            for c in range(c_s, c_e):
                if idx < len(bits):
                    matrix[r, c] = 0 if bits[idx] == '1' else 255
                    idx += 1

    def _fill_data_area(self, matrix, bits):
        mask = self._get_data_area_mask()
        coords = np.argwhere(mask)
        for i, (r, c) in enumerate(coords):
            if i < len(bits):
                matrix[r, c] = 0 if bits[i] == '1' else 255