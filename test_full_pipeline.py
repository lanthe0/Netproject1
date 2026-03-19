"""
完整流程自动化测试脚本
测试不同输入的编码-解码正确率
"""

import os
import sys
import time
import random
import hashlib
import subprocess
from pathlib import Path

# 测试配置
TEST_DIR = Path("test_data")
INPUT_DIR = TEST_DIR / "input"
OUTPUT_DIR = TEST_DIR / "output"
MAX_VIDEO_LENGTH_MS = 50000  # 50秒视频限制（支持更大文件）
ENCODE_TIMEOUT = 300  # 编码超时：5分钟
DECODE_TIMEOUT = 600  # 解码超时：10分钟

# 创建测试目录
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_test_files():
    """生成多种测试文件"""
    test_files = []
    
    # 1. 小文件 - 纯文本 (1KB)
    file1 = INPUT_DIR / "test_text_1kb.txt"
    with open(file1, 'w', encoding='utf-8') as f:
        f.write("Hello World! " * 70)  # ~1KB
    test_files.append(("text_1kb", file1, "text"))
    
    # 2. 小文件 - 随机二进制 (5KB)
    file2 = INPUT_DIR / "test_random_5kb.bin"
    with open(file2, 'wb') as f:
        f.write(os.urandom(5 * 1024))
    test_files.append(("random_5kb", file2, "binary"))
    
    # 3. 中等文件 - 重复模式 (50KB)
    file3 = INPUT_DIR / "test_pattern_50kb.bin"
    pattern = bytes([i % 256 for i in range(256)])
    with open(file3, 'wb') as f:
        f.write(pattern * 200)  # 50KB
    test_files.append(("pattern_50kb", file3, "pattern"))
    
    # 4. 中等文件 - 随机二进制 (100KB)
    file4 = INPUT_DIR / "test_random_100kb.bin"
    with open(file4, 'wb') as f:
        f.write(os.urandom(100 * 1024))
    test_files.append(("random_100kb", file4, "binary"))
    
    # 5. 较大文件 - 随机二进制 (200KB) - 降低以适应测试
    file5 = INPUT_DIR / "test_random_200kb.bin"
    with open(file5, 'wb') as f:
        f.write(os.urandom(200 * 1024))
    test_files.append(("random_200kb", file5, "binary"))
    
    # 6. 大文件 - 随机二进制 (500KB) - 需要更长时间
    file6 = INPUT_DIR / "test_random_500kb.bin"
    with open(file6, 'wb') as f:
        f.write(os.urandom(500 * 1024))
    test_files.append(("random_500kb", file6, "binary"))
    
    # 7. JSON格式文件 (10KB)
    file7 = INPUT_DIR / "test_json_10kb.json"
    with open(file7, 'w', encoding='utf-8') as f:
        data = {"users": [{"id": i, "name": f"user_{i}", "data": "x" * 100} for i in range(100)]}
        import json
        f.write(json.dumps(data, indent=2))
    test_files.append(("json_10kb", file7, "json"))
    
    # 8. 全零文件 (20KB) - 测试压缩场景
    file8 = INPUT_DIR / "test_zeros_20kb.bin"
    with open(file8, 'wb') as f:
        f.write(b'\x00' * (20 * 1024))
    test_files.append(("zeros_20kb", file8, "zeros"))
    
    # 9. 全一文件 (20KB) - 测试压缩场景
    file9 = INPUT_DIR / "test_ones_20kb.bin"
    with open(file9, 'wb') as f:
        f.write(b'\xff' * (20 * 1024))
    test_files.append(("ones_20kb", file9, "ones"))
    
    # 10. 交替模式 (30KB)
    file10 = INPUT_DIR / "test_alternating_30kb.bin"
    with open(file10, 'wb') as f:
        f.write((b'\xaa\x55' * 15) * 1024)
    test_files.append(("alternating_30kb", file10, "alternating"))
    
    return test_files


def calculate_file_hash(filepath):
    """计算文件SHA256哈希"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def run_encode(test_id, input_file, output_video):
    """运行编码过程"""
    cmd = [
        sys.executable,
        "src/encode.py",
        str(input_file),
        str(output_video),
        str(MAX_VIDEO_LENGTH_MS)
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=ENCODE_TIMEOUT,
            encoding='gbk',  # Windows中文编码
            errors='replace'
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            return "success", elapsed, None
        else:
            error_msg = result.stderr or result.stdout
            return "failed", elapsed, error_msg
    except subprocess.TimeoutExpired:
        return "timeout", time.time() - start_time, f"编码超时（超过{ENCODE_TIMEOUT}秒）"
    except Exception as e:
        return "error", time.time() - start_time, str(e)


def run_decode(test_id, input_video, output_bin, output_vbin):
    """运行解码过程"""
    cmd = [
        sys.executable,
        "src/decode.py",
        str(input_video),
        str(output_bin),
        str(output_vbin)
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=DECODE_TIMEOUT,
            encoding='gbk',  # Windows中文编码
            errors='replace'
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            return "success", elapsed, None
        else:
            error_msg = result.stderr or result.stdout
            return "failed", elapsed, error_msg
    except subprocess.TimeoutExpired:
        return "timeout", time.time() - start_time, f"解码超时（超过{DECODE_TIMEOUT}秒）"
    except Exception as e:
        return "error", time.time() - start_time, str(e)


def compare_files(original, decoded):
    """比较原始文件和解码文件"""
    if not os.path.exists(decoded):
        return False, "解码文件不存在"
    
    original_hash = calculate_file_hash(original)
    decoded_hash = calculate_file_hash(decoded)
    
    if original_hash == decoded_hash:
        return True, None
    else:
        # 提供更详细的错误信息
        orig_size = os.path.getsize(original)
        dec_size = os.path.getsize(decoded)
        return False, f"哈希不匹配 (原始:{orig_size}B vs 解码:{dec_size}B)"


def run_test_suite():
    """运行完整测试套件"""
    print("="*80)
    print("开始完整流程测试")
    print("="*80)
    print()
    
    # 生成测试文件
    print("📝 生成测试文件...")
    test_files = generate_test_files()
    print(f"✅ 已生成 {len(test_files)} 个测试文件\n")
    
    results = []
    
    for test_id, input_file, content_type in test_files:
        print(f"\n{'='*60}")
        print(f"🧪 测试: {test_id} ({content_type})")
        print(f"{'='*60}")
        
        file_size = os.path.getsize(input_file)
        print(f"📦 输入文件大小: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        
        # 定义输出路径
        output_video = OUTPUT_DIR / f"{test_id}.mp4"
        output_bin = OUTPUT_DIR / f"{test_id}_decoded.bin"
        output_vbin = OUTPUT_DIR / f"{test_id}_vout.bin"
        
        test_result = {
            'id': test_id,
            'name': test_id,
            'file_size': file_size,
            'content_type': content_type,
            'encode_status': None,
            'decode_status': None,
            'data_match': False,
            'encode_time': 0,
            'decode_time': 0,
            'error_message': None
        }
        
        # 步骤1: 编码
        print(f"\n🔄 [1/3] 编码中...")
        encode_status, encode_time, encode_error = run_encode(test_id, input_file, output_video)
        test_result['encode_status'] = encode_status
        test_result['encode_time'] = encode_time
        
        if encode_status == "success":
            print(f"✅ 编码成功 (耗时: {encode_time:.2f}秒)")
            video_size = os.path.getsize(output_video)
            print(f"   视频文件大小: {video_size:,} bytes ({video_size/1024/1024:.2f} MB)")
        else:
            print(f"❌ 编码失败: {encode_error}")
            test_result['error_message'] = f"编码失败: {encode_error}"
            results.append(test_result)
            continue
        
        # 步骤2: 解码
        print(f"\n🔄 [2/3] 解码中...")
        decode_status, decode_time, decode_error = run_decode(test_id, output_video, output_bin, output_vbin)
        test_result['decode_status'] = decode_status
        test_result['decode_time'] = decode_time
        
        if decode_status == "success":
            print(f"✅ 解码成功 (耗时: {decode_time:.2f}秒)")
        else:
            print(f"❌ 解码失败: {decode_error}")
            test_result['error_message'] = f"解码失败: {decode_error}"
            results.append(test_result)
            continue
        
        # 步骤3: 验证数据
        print(f"\n🔄 [3/3] 验证数据完整性...")
        data_match, verify_error = compare_files(input_file, output_bin)
        test_result['data_match'] = data_match
        
        if data_match:
            print(f"✅ 数据完全匹配！")
        else:
            print(f"❌ 数据不匹配: {verify_error}")
            test_result['error_message'] = verify_error
        
        results.append(test_result)
    
    # 打印统计报告
    print("\n" + "="*80)
    print("📊 测试统计报告")
    print("="*80)
    
    total_tests = len(results)
    encode_success = sum(1 for r in results if r['encode_status'] == 'success')
    decode_success = sum(1 for r in results if r['decode_status'] == 'success')
    data_match_count = sum(1 for r in results if r['data_match'])
    
    print(f"\n总测试数: {total_tests}")
    print(f"编码成功率: {encode_success}/{total_tests} ({encode_success/total_tests*100:.1f}%)")
    print(f"解码成功率: {decode_success}/{total_tests} ({decode_success/total_tests*100:.1f}%)")
    print(f"数据完整性: {data_match_count}/{total_tests} ({data_match_count/total_tests*100:.1f}%)")
    
    # 详细结果表格
    print("\n" + "-"*80)
    print(f"{'测试ID':<20} {'大小':<10} {'编码':<8} {'解码':<8} {'匹配':<6} {'总耗时':<8}")
    print("-"*80)
    
    for r in results:
        size_kb = r['file_size'] / 1024
        encode_mark = '✓' if r['encode_status'] == 'success' else '✗'
        decode_mark = '✓' if r['decode_status'] == 'success' else '✗'
        match_mark = '✓' if r['data_match'] else '✗'
        total_time = r['encode_time'] + r['decode_time']
        
        print(f"{r['id']:<20} {size_kb:>8.1f}KB {encode_mark:<8} {decode_mark:<8} {match_mark:<6} {total_time:>6.2f}s")
    
    print("-"*80)
    
    # 失败详情
    failures = [r for r in results if not r['data_match']]
    if failures:
        print("\n❌ 失败详情:")
        for r in failures:
            print(f"  • {r['id']}: {r['error_message']}")
    else:
        print("\n🎉 所有测试均通过！")
    
    print("\n" + "="*80)
    
    # 保存详细结果到JSON
    import json
    report_file = OUTPUT_DIR / "test_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"📄 详细报告已保存至: {report_file}")
    
    return results


if __name__ == "__main__":
    try:
        results = run_test_suite()
    except KeyboardInterrupt:
        print("\n\n⚠️ 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
