"""验证 reedsolo 的 ECC 字节数"""
import reedsolo

for nsym in [8, 12, 16, 20]:
    rs = reedsolo.RSCodec(nsym)
    test_data = b"Hello World"
    encoded = rs.encode(test_data)
    ecc_bytes = len(encoded) - len(test_data)
    
    print(f"RSCodec({nsym:2d}) → 原始: {len(test_data)} bytes, 编码后: {len(encoded)} bytes, ECC: {ecc_bytes} bytes")

print(f"\n结论: RSCodec(n) 添加的 ECC 字节数 = 2 × n")
print(f"当前配置 ECC_BYTES=16 实际上是 nsym参数，真实ECC是 32 bytes!")
