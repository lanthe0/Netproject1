"""
测试二维码生成功能
"""
import sys
import os
import time

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from _2Dcode import encode_bin, save_test_frames
import numpy as np

def test_encode_small_file():
    """测试编码小文件"""
    print("=" * 60)
    print("测试 1: 编码小文件 (data/test_short.txt)")
    print("=" * 60)
    
    try:
        start_time = time.time()
        grids = encode_bin("data/test_short.txt")
        end_time = time.time()
        
        print(f"✅ 编码成功!")
        print(f"   - 生成帧数: {len(grids)}")
        print(f"   - 矩阵尺寸: {grids[0].shape}")
        print(f"   - 数据类型: {grids[0].dtype}")
        print(f"   - 取值范围: [{grids[0].min()}, {grids[0].max()}]")
        print(f"   - 耗时: {end_time - start_time:.3f} 秒")
        
        # 检查第一帧的定位格
        first_frame = grids[0]
        print(f"\n   检查定位格 (左上角 2,2 位置):")
        print(f"   - 值: {first_frame[2, 2]} (预期: 1 = 黑色)")
        
        return True, grids
    except Exception as e:
        print(f"❌ 编码失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_encode_medium_file():
    """测试编码中等大小文件"""
    print("\n" + "=" * 60)
    print("测试 2: 编码中等文件 (data/test_random.bin - 50KB)")
    print("=" * 60)
    
    try:
        start_time = time.time()
        grids = encode_bin("data/test_random.bin")
        end_time = time.time()
        
        file_size = os.path.getsize("data/test_random.bin")
        
        print(f"✅ 编码成功!")
        print(f"   - 原始文件大小: {file_size:,} bytes")
        print(f"   - 生成帧数: {len(grids)}")
        print(f"   - 每帧理论容量: ~2.3 KB (18748 bits)")
        print(f"   - 理论最小帧数: {file_size * 8 / 18748:.1f}")
        print(f"   - 耗时: {end_time - start_time:.3f} 秒")
        print(f"   - 平均速度: {file_size / (end_time - start_time) / 1024:.1f} KB/s")
        
        return True, grids
    except Exception as e:
        print(f"❌ 编码失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_encode_large_file():
    """测试编码大文件"""
    print("\n" + "=" * 60)
    print("测试 3: 编码大文件 (data/test_pattern.bin - 200KB)")
    print("=" * 60)
    
    try:
        start_time = time.time()
        grids = encode_bin("data/test_pattern.bin")
        end_time = time.time()
        
        file_size = os.path.getsize("data/test_pattern.bin")
        
        print(f"✅ 编码成功!")
        print(f"   - 原始文件大小: {file_size:,} bytes")
        print(f"   - 生成帧数: {len(grids)}")
        print(f"   - 耗时: {end_time - start_time:.3f} 秒")
        print(f"   - 平均速度: {file_size / (end_time - start_time) / 1024:.1f} KB/s")
        
        return True, grids
    except Exception as e:
        print(f"❌ 编码失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_save_frames(grids):
    """测试保存二维码图片"""
    print("\n" + "=" * 60)
    print("测试 4: 保存二维码帧为图片")
    print("=" * 60)
    
    try:
        # 只保存前3帧作为示例
        test_grids = grids[:min(3, len(grids))]
        save_test_frames(test_grids, "output/test_encode_frames")
        
        print(f"✅ 保存成功!")
        print(f"   - 保存位置: output/test_encode_frames/")
        print(f"   - 保存帧数: {len(test_grids)}")
        print(f"   - 文件格式: PNG + NPY")
        
        # 检查文件是否存在
        import glob
        png_files = glob.glob("output/test_encode_frames/*.png")
        npy_files = glob.glob("output/test_encode_frames/*.npy")
        print(f"   - PNG 文件: {len(png_files)} 个")
        print(f"   - NPY 文件: {len(npy_files)} 个")
        
        return True
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n🚀 开始测试二维码生成功能\n")
    
    results = []
    
    # 测试1: 小文件
    success1, grids1 = test_encode_small_file()
    results.append(("小文件编码", success1))
    
    # 测试2: 中等文件
    success2, grids2 = test_encode_medium_file()
    results.append(("中等文件编码", success2))
    
    # 测试3: 大文件
    success3, grids3 = test_encode_large_file()
    results.append(("大文件编码", success3))
    
    # 测试4: 保存帧（使用小文件的结果）
    if success1 and grids1 is not None:
        success4 = test_save_frames(grids1)
        results.append(("保存二维码图片", success4))
    
    # 输出总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status} - {test_name}")
    
    all_passed = all(success for _, success in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过!")
    else:
        print("⚠️  部分测试失败，请检查上述错误信息")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
