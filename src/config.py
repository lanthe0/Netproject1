"""
这个文件用于存放超参数
"""

# 矩阵大小
GRID_SIZE = 108
# 边距
QUIET_WIDTH = 2

# 定位格大小
BIG_FINDER_SIZE = 14
SMALL_FINDER_SIZE = 7

# 区域范围: (row_start, row_end, col_start, col_end)
HEADER_BOUNDS = (2, 10, 18, 26)
SMALL_FINDER_BOUNDS = (99, 106, 99, 106)

# 数据区容量限制
DATA_SIZE_LIMIT = 9884

# 信息头大小
HEADER_SIZE = 64

# 数据区范围
DATA_BOUNDS = (2, 106, 27, 98)

# 每帧增加 16 个 ECC 字节，意味著每帧可以修復 8 个错誤字节
ECC_BYTES = 16 
