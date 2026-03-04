"""
_2Dcode的实现
包含：
1. 编码：将一定长度的二进制数据转换为二维码图片
2. 解码：将二维码图片转换为一定长度的二进制数据与对应的有效性标记
3. 校对：使用opencv根据四角坐标处理偏转和偏移
4. 验证：验证数据和CRC校验码
"""
import numpy as np

class _2Dcode:
    def __init__(self, data, index):
        self.data = np.matrix([])
        
    def to_image(self):
        """
        转为图片
        """
        return 
    

