import sys, cv2, numpy as np
from utils import *

def main():
    assert len(sys.argv) == 4, "缺少参数，正确用法: decode <输入视频文件> <解码后输出的二进制文件> <每位有效性标记输出文件>"
    input_video_path = sys.argv[1]
    output_bin_path = sys.argv[2]
    output_vbin_path = sys.argv[3]
    
    # 使用opencv读取视频文件并转换为图片组
    img_group = []
    
    # 校对图片，截取出二维码，使用opencv根据四角坐标处理偏转和偏移，得到矫正后的二维码图片
    
    # 根据_2Dcode提供的方法对图片进行解码，得到数据和有效性标记，写入对应文件路径
    
    # 结束
    
if __name__ == "__main__":
    main()  
    