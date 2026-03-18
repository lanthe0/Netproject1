"""
详细调试容量计算问题
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from config import ECC_BYTES

# 模拟 encode_bin 的分帧逻辑
print("=" * 60)
print("模拟编码 data/test_random.bin (50KB)")
print("=" * 60)

# 导入后 DATA_SIZE_LIMIT 会被覆盖
from _2Dcode import DATA_SIZE_LIMIT, DATA_ITER, encode_bin, bytes_to_bits
import reedsolo

file_size = os.path.getsize("data/test_random.bin")
print(f"\n1. 文件信息:")
print(f"   - 文件大小: {file_size} bytes = {file_size * 8} bits")

print(f"\n2. 容量配置:")
print(f"   - DATA_SIZE_LIMIT: {DATA_SIZE_LIMIT} bits")
print(f"   - DATA_ITER 长度: {len(DATA_ITER)} bits")
print(f"   - ECC_BYTES: {ECC_BYTES} bytes = {ECC_BYTES * 8} bits")

# 读取文件并转换为 bits
with open("data/test_random.bin", "rb") as f:
    data = f.read()

bit_data = bytes_to_bits(data)
print(f"\n3. 数据转换:")
print(f"   - 总 bit 数: {len(bit_data)}")

# 按 DATA_SIZE_LIMIT 分块
num_chunks = (len(bit_data) + DATA_SIZE_LIMIT - 1) // DATA_SIZE_LIMIT
print(f"   - 预计帧数: {num_chunks}")

# 检查第一帧
first_chunk = bit_data[:DATA_SIZE_LIMIT]
print(f"\n4. 第一帧分析:")
print(f"   - 数据 bits: {len(first_chunk)}")
print(f"   - 数据 bytes: {len(first_chunk) // 8} (实际: {(len(first_chunk) + 7) // 8})")

# 模拟 RS 编码
rs = reedsolo.RSCodec(ECC_BYTES)
raw_bytes = np.packbits(first_chunk).tobytes()
print(f"   - packbits 后: {len(raw_bytes)} bytes")

ecc_payload_bytes = rs.encode(raw_bytes)
print(f"   - RS 编码后: {len(ecc_payload_bytes)} bytes")

ecc_bits = bytes_to_bits(ecc_payload_bytes)
print(f"   - 转回 bits: {len(ecc_bits)} bits")

print(f"\n5. 容量检查:")
print(f"   - 需要容量: {len(ecc_bits)} bits")
print(f"   - 实际容量: {len(DATA_ITER)} bits")
if len(ecc_bits) > len(DATA_ITER):
    print(f"   ❌ 超出: {len(ecc_bits) - len(DATA_ITER)} bits")
    
    # 计算正确的 DATA_SIZE_LIMIT
    max_ecc_bits = len(DATA_ITER)
    max_ecc_bytes = max_ecc_bits // 8
    max_data_bytes = max_ecc_bytes - ECC_BYTES
    max_data_bits = max_data_bytes * 8
    
    print(f"\n6. 修正建议:")
    print(f"   - 最大 ECC payload: {max_ecc_bytes} bytes")
    print(f"   - 减去 ECC: {max_ecc_bytes} - {ECC_BYTES} = {max_data_bytes} bytes")
    print(f"   - 建议 DATA_SIZE_LIMIT: {max_data_bits} bits")
    print(f"   - 当前 DATA_SIZE_LIMIT: {DATA_SIZE_LIMIT} bits")
    print(f"   - 差值: {DATA_SIZE_LIMIT - max_data_bits} bits 过大")
else:
    print(f"   ✅ 容量充足")

print("\n" + "=" * 60)
