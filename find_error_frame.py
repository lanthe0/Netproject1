"""找出具体哪一帧出错"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from _2Dcode import bytes_to_bits, DATA_SIZE_LIMIT, DATA_ITER, ECC_BYTES
import reedsolo

print(f"配置: DATA_SIZE_LIMIT = {DATA_SIZE_LIMIT} bits\n")

with open("data/test_random.bin", "rb") as f:
    data = f.read()

bit_data = bytes_to_bits(data)
print(f"总数据: {len(bit_data)} bits\n")

# 分帧
data_ls = [
    np.array(bit_data[i : i + DATA_SIZE_LIMIT], dtype=np.uint8)
    for i in range(0, len(bit_data), DATA_SIZE_LIMIT)
]

print(f"分帧结果: {len(data_ls)} 帧\n")

rs = reedsolo.RSCodec(ECC_BYTES)

for idx, bits in enumerate(data_ls):
    raw_bytes = np.packbits(bits).tobytes()
    ecc_payload_bytes = rs.encode(raw_bytes)
    ecc_bits_len = len(ecc_payload_bytes) * 8
    
    status = "✅" if ecc_bits_len <= len(DATA_ITER) else "❌"
    print(f"{status} 帧 {idx}: {len(bits)} bits → {len(raw_bytes)} bytes → +ECC → {len(ecc_payload_bytes)} bytes → {ecc_bits_len} bits")
    
    if ecc_bits_len > len(DATA_ITER):
        print(f"   超出: {ecc_bits_len - len(DATA_ITER)} bits")
        break
