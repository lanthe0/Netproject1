import sys, cv2, numpy as np
from _2Dcode import decode_image

def extract_frames_from_video(video_path: str) -> list:
    """
    从视频文件中提取帧序列
    Args:
        video_path: 视频文件路径
    Returns:
        帧列表，每个帧是一个 numpy 数组
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件：{video_path}")
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"--- 视频信息 ---")
    print(f"视频路径：{video_path}")
    print(f"帧率：{fps} fps")
    print(f"总帧数：{frame_count}")
    print(f"分辨率：{width}x{height}")
    print(f"----------------")
    
    print(f"正在从视频中提取帧...")
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        extracted_count += 1
        
        # 每提取 10 帧显示一次进度
        if extracted_count % 10 == 0:
            print(f"已提取 {extracted_count}/{frame_count} 帧 ({extracted_count/frame_count*100:.1f}%)")
    
    cap.release()
    print(f"视频帧提取完成，共提取 {len(frames)} 帧")
    
    return frames

def main():
    """
    解码器主函数
    从视频中提取帧序列，然后调用 decode_image 进行解码
    
    Args:
        sys.argv[1]: 输入视频文件路径
        sys.argv[2]: 解码后输出的二进制文件路径
        sys.argv[3]: 每位有效性标记输出文件路径
    """
    # 参数校验
    if len(sys.argv) != 4:
        print("Usage: python decode.py <input_video> <output_bin> <output_vbin>")
        print("Example: python decode.py output.mp4 output/decoded.bin output/vout.bin")
        sys.exit(1)
    
    input_video_path = sys.argv[1]
    output_bin_path = sys.argv[2]
    output_vbin_path = sys.argv[3]
    
    # 1. 检查输入文件是否存在
    import os
    if not os.path.exists(input_video_path):
        print(f"Error: 找不到输入文件 '{input_video_path}'")
        sys.exit(1)
    
    # 2. 从视频中提取帧序列
    try:
        frames = extract_frames_from_video(input_video_path)
    except Exception as e:
        print(f"Error: 提取视频帧失败 - {e}")
        sys.exit(1)
    
    if not frames:
        print("Error: 未从视频中提取到任何帧")
        sys.exit(1)
    
    # 3. 调用 decode_image 进行解码
    print(f"\n--- 开始解码 ---")
    print(f"输入帧数：{len(frames)}")
    print(f"输出二进制文件：{output_bin_path}")
    print(f"输出有效性标记文件：{output_vbin_path}")
    
    try:
        decode_image(frames, output_bin_path, output_vbin_path)
        print(f"\n--- 解码完成 ---")
        print(f"解码成功！")
    except Exception as e:
        print(f"Error: 解码失败 - {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()  
