"""
这个文件用于存放超参数
"""

# 矩阵大小
GRID_SIZE = 108
# 边距
QUIET_WIDTH = 2

# 定位格大小
BIG_FINDER_SIZE = 14
SMALL_FINDER_SIZE = 8

# 区域范围: (row_start, row_end, col_start, col_end)
HEADER_BOUNDS = (2, 16, 18, 90)   # 14 x 72
CHECK_BOUNDS = (18, 90, 2, 16)    # 72 x 14
DATA_BOUNDS = (18, 106, 18, 106)  # 88 x 88
SMALL_FINDER_BOUNDS = (98, 106, 98, 106)  # 8 x 8, inside data area

# 数据区容量限制
DATA_SIZE_LIMIT = 88 * 88 - 64
