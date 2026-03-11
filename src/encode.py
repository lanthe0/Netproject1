import sys
import os
import time
import numpy as np
import cv2
from utils.bin2video import bin2video
from utils.showimg import *
from _2Dcode import encode_bin

def main():
    
    """
    Args:
        input_file_path: 输入二进制文件路径，长度 ≤ 10MB
        output_file_path: 输出视频文件路径
        max_length_of_video: 视频最大长度，单位：毫秒ms
    """
    #vassert len(sys.argv) == 4, "参数格式错误，正确用法: encode <input_file> <output_file> <max_length_of_video>"
    # 1. 基本参数校验
    if len(sys.argv) != 4:
        print("Usage: python encode.py <input_file> <output_file> <max_length_ms>")
        print("Example: python encode.py test.bin output.mp4 10000")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    try:
        max_length_of_video = int(sys.argv[3])
    except ValueError:
        print("Error: max_length_of_video 必须是一个整数（毫秒）。")
        sys.exit(1)
        
    # 2. 文件存在性校验
    if not os.path.exists(input_file_path):
        print(f"Error: 找不到输入文件 '{input_file_path}'")
        sys.exit(1)
    
    # 3. 文件大小校验
    if  os.path.getsize(input_file_path) > 10 * 1024 * 1024:
        print("输入文件大小超过10MB")
        sys.exit(1)
        
    # 3. 统计编码耗时
    start_time = time.time()
    
    print(f"--- 启动编码流水线 ---")
    print(f"输入文件: {input_file_path} ({os.path.getsize(input_file_path)} bytes)")
    print(f"目标路径: {output_file_path}")
    print(f"时长限制: {max_length_of_video} ms")
    
    # 文件转视频
    try:
        # 执行核心 bin2video 逻辑
        grids = encode_bin(input_file_path)
        # grids = matrix_to_bw_image(grids, pixel_per_cell=1)  # 转为图片格式
        bin2video(grids=grids, output_path=output_file_path, max_length_of_video=max_length_of_video)
        # for grid in grids:
        #     show_binary_matrix(grid, pixel_per_cell=10, window_name="Sample Frame", wait_ms=1000)
        end_time = time.time()
        print(f"--- 编码成功 ---")
        print(f"总耗时: {end_time - start_time:.2f} 秒")
        print(f"生成视频位置: {os.path.abspath(output_file_path)}")
    
    except AssertionError as e:
        # 捕获 bin2video 中关于文件大小或时长超限的断言错误
        print(f"编码中止: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"发生意外错误: {e}")
        sys.exit(1)
        
    
    
if __name__ == "__main__":
    main()
