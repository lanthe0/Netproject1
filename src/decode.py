import sys, cv2, numpy as np
from utils import *

def main():
    assert len(sys.argv) == 4, "缺少参数，正确用法: decode <输入视频文件> <解码后输出的二进制文件> <每位有效性标记输出文件>"
    input_video_path = sys.argv[1]
    output_bin_path = sys.argv[2]
    output_vbin_path = sys.argv[3]
    
    
    
if __name__ == "__main__":
    main()  
    