import numpy as np
from _2Dcode import encode_bin, decode_image

def test_encode_decode_cycle(input_file_path: str):
    print(f"--- 開始測試: {input_file_path} ---")
    
    # 1. 編碼: 獲得矩陣序列
    try:
        grids = encode_bin(input_file_path)
        print(f"編碼成功，生成幀數: {len(grids)}")
    except Exception as e:
        print(f"編碼失敗: {e}")
        return

    # 2. 準備路徑
    decoded_bin = "output/decoded_test.bin"
    validity_bin = "output/vout_test.bin"
    
    # 3. 解碼: 將矩陣序列還原為文件
    # 注意: 這裡直接傳入 grids，因為你的 decode_image 函數支援 ndarray 列表
    decode_image(grids, decoded_bin, validity_bin)
    print("解碼完成，正在比對文件...")

    # 4. 比對: 確保編碼前的數據與解碼後的數據完全一致
    with open(input_file_path, "rb") as f1, open(decoded_bin, "rb") as f2:
        if f1.read() == f2.read():
            print("✅ 測試通過！原始數據與解碼數據完全一致。")
        else:
            print("❌ 測試失敗！數據存在差異。")

# 運行測試
test_encode_decode_cycle("data/test_pattern.bin")