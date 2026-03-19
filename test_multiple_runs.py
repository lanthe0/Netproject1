# -*- coding: utf-8 -*-
"""
多轮次测试脚本 - 运行20次完整测试
每轮测试所有文件，汇总统计结果
"""

import os
import sys
import time
import hashlib
import subprocess
import json
from pathlib import Path
from datetime import datetime

# 设置标准输出为UTF-8编码（Windows兼容）
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 测试配置
TEST_DIR = Path("test_data")
INPUT_DIR = TEST_DIR / "input"
OUTPUT_DIR = TEST_DIR / "output"
MAX_VIDEO_LENGTH_MS = 50000  # 50秒视频限制
ENCODE_TIMEOUT = 300  # 编码超时：5分钟
DECODE_TIMEOUT = 600  # 解码超时：10分钟
NUM_RUNS = 20  # 测试轮数

# 创建测试目录
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_test_files():
    """获取已存在的测试文件列表"""
    test_files = []
    
    file_list = [
        ("text_1kb", "test_text_1kb.txt", "text"),
        ("random_5kb", "test_random_5kb.bin", "binary"),
        ("pattern_50kb", "test_pattern_50kb.bin", "pattern"),
        ("random_100kb", "test_random_100kb.bin", "binary"),
        ("random_200kb", "test_random_200kb.bin", "binary"),
        ("json_10kb", "test_json_10kb.json", "json"),
        ("zeros_20kb", "test_zeros_20kb.bin", "zeros"),
        ("ones_20kb", "test_ones_20kb.bin", "ones"),
        ("alternating_30kb", "test_alternating_30kb.bin", "alternating"),
    ]
    
    for test_id, filename, content_type in file_list:
        filepath = INPUT_DIR / filename
        if filepath.exists():
            test_files.append((test_id, filepath, content_type))
    
    return test_files


def calculate_file_hash(filepath):
    """计算文件SHA256哈希"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def run_encode(test_id, input_file, output_video, run_num):
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
            encoding='utf-8',
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


def run_decode(test_id, input_video, output_bin, output_vbin, run_num):
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
            encoding='utf-8',
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
        orig_size = os.path.getsize(original)
        dec_size = os.path.getsize(decoded)
        return False, f"哈希不匹配 (原始:{orig_size}B vs 解码:{dec_size}B)"


def run_single_test(test_id, input_file, content_type, run_num):
    """运行单个文件的测试"""
    file_size = os.path.getsize(input_file)
    
    # 定义输出路径（包含轮次编号）
    output_video = OUTPUT_DIR / f"{test_id}_run{run_num}.mp4"
    output_bin = OUTPUT_DIR / f"{test_id}_run{run_num}_decoded.bin"
    output_vbin = OUTPUT_DIR / f"{test_id}_run{run_num}_vout.bin"
    
    test_result = {
        'run': run_num,
        'id': test_id,
        'name': test_id,
        'file_size': file_size,
        'content_type': content_type,
        'encode_status': None,
        'decode_status': None,
        'data_match': False,
        'encode_time': 0,
        'decode_time': 0,
        'error_message': None,
        'timestamp': datetime.now().isoformat()
    }
    
    # 步骤1: 编码
    encode_status, encode_time, encode_error = run_encode(test_id, input_file, output_video, run_num)
    test_result['encode_status'] = encode_status
    test_result['encode_time'] = encode_time
    
    if encode_status != "success":
        test_result['error_message'] = f"编码失败: {encode_error}"
        return test_result
    
    # 步骤2: 解码
    decode_status, decode_time, decode_error = run_decode(test_id, output_video, output_bin, output_vbin, run_num)
    test_result['decode_status'] = decode_status
    test_result['decode_time'] = decode_time
    
    if decode_status != "success":
        test_result['error_message'] = f"解码失败: {decode_error}"
        return test_result
    
    # 步骤3: 验证数据
    data_match, verify_error = compare_files(input_file, output_bin)
    test_result['data_match'] = data_match
    
    if not data_match:
        test_result['error_message'] = verify_error
    
    # 清理临时文件以节省空间
    try:
        if output_video.exists():
            output_video.unlink()
        if output_bin.exists():
            output_bin.unlink()
        if output_vbin.exists():
            output_vbin.unlink()
    except:
        pass
    
    return test_result


def run_multi_test():
    """运行多轮测试"""
    print("=" * 80)
    print(f"开始 {NUM_RUNS} 轮测试")
    print("=" * 80)
    print(f"配置: 视频时长限制={MAX_VIDEO_LENGTH_MS}ms, 编码超时={ENCODE_TIMEOUT}s, 解码超时={DECODE_TIMEOUT}s")
    print()
    
    # 获取测试文件
    test_files = get_test_files()
    if not test_files:
        print("错误: 未找到测试文件，请先运行 test_pipeline_fixed.py 生成测试文件")
        return
    
    print(f"[*] 找到 {len(test_files)} 个测试文件")
    print(f"[*] 将进行 {NUM_RUNS} 轮测试，总计 {len(test_files) * NUM_RUNS} 个测试案例")
    print()
    
    all_results = []
    
    start_time_total = time.time()
    
    for run_num in range(1, NUM_RUNS + 1):
        print(f"\n{'=' * 80}")
        print(f"第 {run_num}/{NUM_RUNS} 轮测试")
        print(f"{'=' * 80}")
        
        run_results = []
        
        for idx, (test_id, input_file, content_type) in enumerate(test_files, 1):
            file_size = os.path.getsize(input_file)
            print(f"  [{idx}/{len(test_files)}] {test_id} ({file_size/1024:.1f}KB)...", end=" ", flush=True)
            
            result = run_single_test(test_id, input_file, content_type, run_num)
            run_results.append(result)
            
            # 简短输出状态
            if result['data_match']:
                print(f"OK (E:{result['encode_time']:.1f}s D:{result['decode_time']:.1f}s)")
            else:
                print(f"FAIL ({result['error_message'][:30]}...)")
        
        # 统计本轮结果
        success_count = sum(1 for r in run_results if r['data_match'])
        print(f"\n  本轮成功: {success_count}/{len(test_files)} ({success_count/len(test_files)*100:.1f}%)")
        
        all_results.extend(run_results)
    
    elapsed_total = time.time() - start_time_total
    
    # 生成统计报告
    print("\n" + "=" * 80)
    print("测试完成 - 统计报告")
    print("=" * 80)
    
    total_tests = len(all_results)
    encode_success = sum(1 for r in all_results if r['encode_status'] == 'success')
    decode_success = sum(1 for r in all_results if r['decode_status'] == 'success')
    data_match_count = sum(1 for r in all_results if r['data_match'])
    
    print(f"\n总测试数: {total_tests}")
    print(f"总测试时间: {elapsed_total/60:.1f} 分钟")
    print(f"编码成功率: {encode_success}/{total_tests} ({encode_success/total_tests*100:.2f}%)")
    print(f"解码成功率: {decode_success}/{total_tests} ({decode_success/total_tests*100:.2f}%)")
    print(f"数据完整性: {data_match_count}/{total_tests} ({data_match_count/total_tests*100:.2f}%)")
    
    # 按文件统计
    print("\n" + "-" * 80)
    print("分文件统计:")
    print("-" * 80)
    
    file_stats = {}
    for result in all_results:
        test_id = result['id']
        if test_id not in file_stats:
            file_stats[test_id] = {
                'total': 0,
                'success': 0,
                'encode_times': [],
                'decode_times': [],
                'file_size': result['file_size']
            }
        
        file_stats[test_id]['total'] += 1
        if result['data_match']:
            file_stats[test_id]['success'] += 1
        if result['encode_status'] == 'success':
            file_stats[test_id]['encode_times'].append(result['encode_time'])
        if result['decode_status'] == 'success':
            file_stats[test_id]['decode_times'].append(result['decode_time'])
    
    print(f"{'文件ID':<20} {'大小':<10} {'成功率':<15} {'平均编码':<12} {'平均解码':<12}")
    print("-" * 80)
    
    for test_id, stats in sorted(file_stats.items(), key=lambda x: x[1]['file_size']):
        success_rate = stats['success'] / stats['total'] * 100
        avg_encode = sum(stats['encode_times']) / len(stats['encode_times']) if stats['encode_times'] else 0
        avg_decode = sum(stats['decode_times']) / len(stats['decode_times']) if stats['decode_times'] else 0
        size_kb = stats['file_size'] / 1024
        
        print(f"{test_id:<20} {size_kb:>8.1f}KB {stats['success']}/{stats['total']} ({success_rate:>5.1f}%) {avg_encode:>10.2f}s {avg_decode:>10.2f}s")
    
    print("-" * 80)
    
    # 保存详细结果到JSON
    report_file = OUTPUT_DIR / "test_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[*] 详细报告已保存至: {report_file}")
    print(f"[*] 共 {len(all_results)} 条测试记录")
    
    # 生成简要统计文件
    summary = {
        'test_date': datetime.now().isoformat(),
        'num_runs': NUM_RUNS,
        'total_tests': total_tests,
        'total_time_minutes': elapsed_total / 60,
        'encode_success_rate': encode_success / total_tests * 100,
        'decode_success_rate': decode_success / total_tests * 100,
        'data_integrity_rate': data_match_count / total_tests * 100,
        'file_statistics': {}
    }
    
    for test_id, stats in file_stats.items():
        summary['file_statistics'][test_id] = {
            'file_size_kb': stats['file_size'] / 1024,
            'success_rate': stats['success'] / stats['total'] * 100,
            'avg_encode_time': sum(stats['encode_times']) / len(stats['encode_times']) if stats['encode_times'] else 0,
            'avg_decode_time': sum(stats['decode_times']) / len(stats['decode_times']) if stats['decode_times'] else 0,
            'total_runs': stats['total'],
            'successful_runs': stats['success']
        }
    
    summary_file = OUTPUT_DIR / "test_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"[*] 统计摘要已保存至: {summary_file}")
    
    return all_results


if __name__ == "__main__":
    try:
        results = run_multi_test()
    except KeyboardInterrupt:
        print("\n\n[!] 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[-] 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
