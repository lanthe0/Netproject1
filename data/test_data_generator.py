import os
import random

def generate_test_files():
    """
    為 NetProject1 生成三種層次的測試數據
    """
    os.makedirs('data', exist_ok=True)

    # 1. 短文本測試 (驗證基礎封裝與 Header)
    # 特點：數據量小，只有 1 幀
    with open('data/test_short.txt', 'w', encoding='utf-8') as f:
        f.write("Hello Optical Wireless Communication! This is Frame 0 test.")
    print("生成成功: data/test_short.txt (短文本)")

    # 2. 隨機二進位數據 (驗證位對齊與完整性)
    # 特點：數據隨機，容易發現 Bit Offset 錯誤。生成約 50KB。
    with open('input/test_random.bin', 'wb') as f:
        f.write(os.urandom(50 * 1024))
    print("生成成功: data/test_random.bin (50KB 随机数据)")

    # 3. 圖片文件 (驗證多幀重組與傳輸能力)
    # 特點：數據較大，會生成多幀視頻。
    # 如果你有一個現成的小圖片，也可以直接複製過去。
    with open('data/test_pattern.bin', 'wb') as f:
        # 生成一個 200KB 的規律數據，模擬中型文件
        pattern = bytes([i % 256 for i in range(200 * 1024)])
        f.write(pattern)
    print("生成成功: data/test_pattern.bin (200KB 规律数据)")

if __name__ == "__main__":
    generate_test_files()