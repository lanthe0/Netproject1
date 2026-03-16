from utils.bin2img import bin2img
import cv2
from config import *
from utils.showimg import *
import numpy as np
    
def bin2video(grids: np.ndarray, output_path: str, max_length_of_video: int, fps: int = 15):
    """
    将二进制文件转换为视频文件, 视频格式为mp4, 写入output_path
    Args:
        path: 输入二进制文件路径，长度 ≤ 10MB
        output_path: 输出视频文件路径
        max_length_of_video: 视频最大长度，单位：毫秒ms
    """   
  
    total_frames = grids.shape[0]
    
    # 校验时长
    required_ms = (total_frames / fps) * 1000
    assert required_ms <= max_length_of_video, f"当前时长限制 ({max_length_of_video}ms) 无法容纳数据，需要约 {int(required_ms)}ms"
        
    # 1. 将二进制数据转换为图片组
    img_group = []
    for i in range(total_frames):
        img_group.append(matrix_to_bw_image(grids[i], pixel_per_cell=1))
    
    # 2. 使用opencv将img_group写入视频文件
    # 视频参数
    # 为了拍摄清晰，建议将 108x108 放大，例如放大 10 倍到 1080x1080
    upscale = 10
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (GRID_SIZE * upscale, GRID_SIZE * upscale))
    
    for img in img_group:
        # 使用 INTER_NEAREST 保证像素点边缘锐利，不模糊
        frame = cv2.resize(img, (GRID_SIZE * upscale, GRID_SIZE * upscale), interpolation=cv2.INTER_NEAREST)
        # 转为 BGR 格式
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)
        
    out.release()
    print(f"成功生成视频: {output_path}, 共 {total_frames} 帧")