import bin2img
import cv2
from config import *

def bin2video(path: str, output_path: str, max_length_of_video: int, fps: int = 15):
    """
    将二进制文件转换为视频文件, 视频格式为mp4, 写入output_path\n
    Args:
        path: 输入二进制文件路径，长度 ≤ 10MB
        output_path: 输出视频文件路径
        max_length_of_video: 视频最大长度，单位：毫秒ms
    """
    
    with open(path, "rb") as f:
        data = f.read()
        assert len(data) <= 10 * 1024 * 1024, "输入文件大小超过10MB"
        assert len(data) <= DATA_SIZE_LIMIT * (max_length_of_video * fps // 1000), "当前时长限制无法生成合法视频，请调大限制！"
        
    # 将二进制数据转换为图片组
    img_group = bin2img(data)
    
    # 使用opencv将img_group写入视频文件
    
    # 待补充