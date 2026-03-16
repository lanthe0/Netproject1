import reedsolo

rs = reedsolo.RSCodec(10) # 10个冗余字节
# 编码测试
data = b"hello xmu"
encoded = rs.encode(data)
# 模拟损坏
corrupted = list(encoded)
corrupted[0] = 0  # 破坏一个字节
# 尝试修复
decoded = rs.decode(bytes(corrupted))[0]
print(decoded) # 依然能输出 b"hello xmu"