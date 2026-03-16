import numpy as np
from _2Dcode import encode_bin, decode_image

def test_encode_decode_cycle(input_file_path: str):
    print(f"--- 开始测试: {input_file_path} ---")
    
    # 1. 编码: 获取矩阵序列
    try:
        grids = encode_bin(input_file_path)
        print(f"编码成功，生成帧数: {len(grids)}")
    except Exception as e:
        print(f"编码失败: {e}")
        return

    # 2. 准备路径
    decoded_bin = "output/decoded_test.bin"
    validity_bin = "output/vout_test.bin"
    
    # 3. 解码: 将矩阵序列还原为文件
    # 注意: 这这里直接传入 grids，因为你的 decode_image 函数支持 ndarray 列表
    print("解码完成，正在比对文件...")

    # 4. 比对: 确保编码前的数数据与解码后的数数据完全一致   

    with open(input_file_path, "rb") as f1, open(decoded_bin, "rb") as f2:
        if f1.read() == f2.read():
            print("✅ 测试通过！原始数据与解码数据完全一致。")
        else:
            print("❌ 测试失败！数据存在差异。")
# 运行测试
test_encode_decode_cycle("data/test_pattern.bin")