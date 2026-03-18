"""
分析二维码容量问题
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import ECC_BYTES, DATA_SIZE_LIMIT
from _2Dcode import DATA_ITER
import numpy as np

print("=" * 60)
print("二维码容量分析")
print("=" * 60)

print(f"\n配置参数:")
print(f"  - ECC_BYTES (纠错码字节数): {ECC_BYTES}")
print(f"  - DATA_SIZE_LIMIT (数据区容量): {DATA_SIZE_LIMIT} bits")
print(f"  - DATA_ITER 长度: {len(DATA_ITER)} 个位置")

print(f"\n容量计算:")
# 每帧的纯数据容量（不含ECC）
pure_data_bits = DATA_SIZE_LIMIT
pure_data_bytes = pure_data_bits // 8

print(f"  - 纯数据容量: {pure_data_bits} bits = {pure_data_bytes} bytes")

# 加上 ECC 后的总长度
total_bytes = pure_data_bytes + ECC_BYTES
total_bits = total_bytes * 8

print(f"  - 加上ECC后: {total_bytes} bytes = {total_bits} bits")
print(f"  - 二维码容量: {len(DATA_ITER)} bits")

print(f"\n问题诊断:")
if total_bits > len(DATA_ITER):
    overflow = total_bits - len(DATA_ITER)
    print(f"  ❌ 容量不足!")
    print(f"     需要: {total_bits} bits")
    print(f"     可用: {len(DATA_ITER)} bits")
    print(f"     溢出: {overflow} bits ({overflow // 8} bytes)")
    
    # 计算实际能用的数据量
    max_total_bytes = len(DATA_ITER) // 8
    max_data_bytes = max_total_bytes - ECC_BYTES
    max_data_bits = max_data_bytes * 8
    
    print(f"\n  建议修正:")
    print(f"     DATA_SIZE_LIMIT 应设为: {max_data_bits} bits")
    print(f"     这样每帧可存: {max_data_bytes} bytes 数据 + {ECC_BYTES} bytes ECC")
else:
    print(f"  ✅ 容量充足")

print("\n" + "=" * 60)
