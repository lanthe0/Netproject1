"""测试 packbits 的行为"""
import numpy as np

print("测试 np.packbits 行为:\n")

# 测试不同长度的 bits
for bits_len in [18616, 18615, 18617, 18620, 18624]:
    bits = np.ones(bits_len, dtype=np.uint8)
    packed = np.packbits(bits).tobytes()
    print(f"  {bits_len} bits → packbits → {len(packed)} bytes (理论: {(bits_len + 7) // 8})")

print("\n问题分析:")
print(f"  18616 bits ÷ 8 = {18616 / 8} = 2327 bytes")
print(f"  18616 bits % 8 = {18616 % 8} (余数)")
print(f"  packbits 会填充到: {(18616 + 7) // 8} bytes = 2327 bytes")
print(f"  加上 ECC 16 bytes = {2327 + 16} = 2343 bytes")
print(f"  转回 bits = {2343 * 8} = 18744 bits")
print(f"  容量限制 = 18748 bits")
print(f"  ✅ 没问题！")

print("\n但为什么还是报错 19896 bits?")
print(f"  19896 ÷ 8 = {19896 / 8} = 2487 bytes")
print(f"  2487 - 16 (ECC) = 2471 bytes")
print(f"  2471 × 8 = 19768 bits")
print(f"  这不等于 18616...")

print("\n让我计算最后一帧的情况:")
total_bits = 409600  # 50KB file
bits_per_frame = 18616
num_full_frames = total_bits // bits_per_frame
last_frame_bits = total_bits % bits_per_frame

print(f"  总 bits: {total_bits}")
print(f"  每帧: {bits_per_frame} bits")
print(f"  完整帧: {num_full_frames}")
print(f"  最后一帧: {last_frame_bits} bits")
print(f"  最后一帧 packbits: {(last_frame_bits + 7) // 8} bytes")
print(f"  加 ECC: {(last_frame_bits + 7) // 8 + 16} bytes")
print(f"  转 bits: {((last_frame_bits + 7) // 8 + 16) * 8} bits")
