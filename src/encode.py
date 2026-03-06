import sys
import numpy as np
import cv2
from utils.bin2video import bin2video

def main():
    
    """
    Args:
        input_file_path: 输入二进制文件路径，长度 ≤ 10MB
        output_file_path: 输出视频文件路径
        max_length_of_video: 视频最大长度，单位：毫秒ms
    """
    assert len(sys.argv) == 4, "缺少参数，正确用法: encode <input_file> <output_file> <max_length_of_video>"
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    max_length_of_video = int(sys.argv[3])
    
    # 文件转视频
    bin2video(path=input_file_path, output_path=output_file_path, max_length_of_video=max_length_of_video)
    
    
if __name__ == "__main__":
    main()
