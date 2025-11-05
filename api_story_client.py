#!/usr/bin/env python3
"""
故事音频生成 API 自动测试工具

自动运行测试用例，验证 /generate_story_audio API 功能
"""

import requests
import json
import base64
import argparse
from pathlib import Path
import time
from datetime import datetime

USE_LOCAL = False
BASE_URL = "http://localhost:6006" if USE_LOCAL else "http://117.50.190.136:6006"


# 测试用例配置
TEST_CASES = [
    # {
    #     "name": "无音色映射测试（文件上传）",
    #     "description": "测试使用默认音色映射生成故事音频（文件上传方式）",
    #     "json_file": "examples/role_778_True.json",
    #     "voice_map": None,
    #     "output_suffix": "no_voicemap",
    #     "use_json_api": False
    # },
    # {
    #     "name": "无音色映射测试（JSON直接调用）",
    #     "description": "测试使用默认音色映射生成故事音频（JSON直接调用方式）",
    #     "json_file": "examples/role_778_True.json",
    #     "voice_map": None,
    #     "output_suffix": "json_api_no_voicemap",
    #     "use_json_api": True
    # },
    {
        "name": "单个条目批量测试",
        "description": "依次调用 generate_story_audio_single API 处理前N个条目",
        "json_file": "examples/role_778_True.json",
        "item_count": 3,  # 处理前5个条目
        "output_suffix": "single_batch",
        "use_single_api": True
    },
    # {
    #     "name": "自定义音色映射测试",
    #     "description": "测试使用自定义音色映射生成故事音频",
    #     "json_file": "examples/role_778_True.json",
    #     "voice_map": "examples/voice_map_example.json",
    #     "output_suffix": "with_voicemap",
    #     "use_json_api": False
    # }
]


def generate_story_audio(
    api_url: str, 
    json_file_path: str, 
    output_dir: str = "outputs",
    voice_map_path: str = None,
    case_name: str = "test"
):
    """
    调用 API 生成故事音频和字幕
    
    Args:
        api_url: API 服务器地址
        json_file_path: 故事 JSON 文件路径
        output_dir: 输出目录
        voice_map_path: 自定义音色映射 JSON 文件路径（可选）
        case_name: 测试用例名称（用于输出文件命名）
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 读取 JSON 文件
    print(f">> 读取 JSON 文件: {json_file_path}")
    
    # 准备请求数据
    files = {}
    data = {}
    
    with open(json_file_path, 'rb') as f:
        files['file'] = (Path(json_file_path).name, f, 'application/json')
        
        # 如果提供了自定义音色映射
        if voice_map_path and Path(voice_map_path).exists():
            print(f">> 使用自定义音色映射: {voice_map_path}")
            with open(voice_map_path, 'r', encoding='utf-8') as vm:
                voice_map_content = vm.read()
                data['voice_map'] = voice_map_content
        
        # 调用 API
        print(f">> 调用 API: {api_url}/generate_story_audio")
        print(">> 正在生成音频和字幕，请耐心等待...")
        
        try:
            response = requests.post(
                f"{api_url}/generate_story_audio",
                files=files,
                data=data,
                timeout=6000  # 100分钟超时
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("status") == "success":
                    print(f"\n>> ✓ 成功!")
                    print(f"   - 处理条目: {result['processed_items']}")
                    print(f"   - 跳过条目: {result['skipped_items']}")
                    print(f"   - 总时长: {result['total_duration']/1000:.2f} 秒")
                    print(f"   - 字幕数量: {result['subtitle_count']}")
                    print(f"   - 采样率: {result['sample_rate']} Hz")
                    
                    # 保存音频文件
                    audio_base64 = result['audio']
                    audio_bytes = base64.b64decode(audio_base64)
                    
                    # 生成输出文件名（使用测试用例名称）
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    audio_output_path = Path(output_dir) / f"{case_name}_{timestamp}.wav"
                    subtitle_output_path = Path(output_dir) / f"{case_name}_{timestamp}.json"
                    subtitle_srt_path = Path(output_dir) / f"{case_name}_{timestamp}.srt"
                    
                    # 保存音频
                    with open(audio_output_path, 'wb') as f:
                        f.write(audio_bytes)
                    print(f"   - 音频已保存: {audio_output_path}")
                    
                    # 保存字幕（JSON 格式）
                    subtitles = result['subtitles']
                    with open(subtitle_output_path, 'w', encoding='utf-8') as f:
                        json.dump(subtitles, f, ensure_ascii=False, indent=2)
                    print(f"   - 字幕已保存: {subtitle_output_path}")
                    
                    # 保存字幕（SRT 格式）
                    generate_srt(subtitles, subtitle_srt_path)
                    print(f"   - SRT 字幕已保存: {subtitle_srt_path}")
                    
                    print(f"\n>> 所有文件已保存到: {output_dir}/")
                    return True
                else:
                    print(f"\n>> ✗ API 返回错误: {result.get('error', '未知错误')}")
                    return False
            else:
                print(f"\n>> ✗ HTTP 错误: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   错误信息: {error_data.get('error', response.text)}")
                except:
                    print(f"   响应内容: {response.text[:500]}")
                return False
                
        except requests.exceptions.Timeout:
            print("\n>> ✗ 请求超时，故事可能太长，请尝试分段处理")
            return False
        except requests.exceptions.ConnectionError:
            print(f"\n>> ✗ 无法连接到 API 服务器: {api_url}")
            print("   请确保服务器正在运行")
            return False
        except Exception as ex:
            print(f"\n>> ✗ 发生错误: {str(ex)}")
            import traceback
            traceback.print_exc()
            return False


def generate_story_audio_json_api(
    api_url: str,
    json_file_path: str,
    output_dir: str = "outputs",
    voice_map_path: str = None,
    case_name: str = "test"
):
    """
    调用 JSON API 生成故事音频和字幕（直接传递 JSON 数据，不上传文件）
    
    Args:
        api_url: API 服务器地址
        json_file_path: 故事 JSON 文件路径
        output_dir: 输出目录
        voice_map_path: 自定义音色映射 JSON 文件路径（可选）
        case_name: 测试用例名称（用于输出文件命名）
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 读取 JSON 文件
    print(f">> 读取 JSON 文件: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        story_data = json.load(f)
    
    # 准备请求数据
    request_body = {
        "story_data": story_data
    }
    
    # 如果提供了自定义音色映射
    if voice_map_path and Path(voice_map_path).exists():
        print(f">> 使用自定义音色映射: {voice_map_path}")
        with open(voice_map_path, 'r', encoding='utf-8') as vm:
            voice_map = json.load(vm)
            request_body['voice_map'] = voice_map
    
    # 调用 API
    print(f">> 调用 API: {api_url}/generate_story_audio_json")
    print(">> 正在生成音频和字幕，请耐心等待...")
    
    try:
        response = requests.post(
            f"{api_url}/generate_story_audio_json",
            json=request_body,
            headers={"Content-Type": "application/json"},
            timeout=6000  # 100分钟超时
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("status") == "success":
                print(f"\n>> ✓ 成功!")
                print(f"   - 处理条目: {result['processed_items']}")
                print(f"   - 跳过条目: {result['skipped_items']}")
                print(f"   - 总时长: {result['total_duration']/1000:.2f} 秒")
                print(f"   - 字幕数量: {result['subtitle_count']}")
                print(f"   - 采样率: {result['sample_rate']} Hz")
                
                # 保存音频文件
                audio_base64 = result['audio']
                audio_bytes = base64.b64decode(audio_base64)
                
                # 生成输出文件名（使用测试用例名称）
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_output_path = Path(output_dir) / f"{case_name}_{timestamp}.wav"
                subtitle_output_path = Path(output_dir) / f"{case_name}_{timestamp}.json"
                subtitle_srt_path = Path(output_dir) / f"{case_name}_{timestamp}.srt"
                
                # 保存音频
                with open(audio_output_path, 'wb') as f:
                    f.write(audio_bytes)
                print(f"   - 音频已保存: {audio_output_path}")
                
                # 保存字幕（JSON 格式）
                subtitles = result['subtitles']
                with open(subtitle_output_path, 'w', encoding='utf-8') as f:
                    json.dump(subtitles, f, ensure_ascii=False, indent=2)
                print(f"   - 字幕已保存: {subtitle_output_path}")
                
                # 保存字幕（SRT 格式）
                generate_srt(subtitles, subtitle_srt_path)
                print(f"   - SRT 字幕已保存: {subtitle_srt_path}")
                
                print(f"\n>> 所有文件已保存到: {output_dir}/")
                return True
            else:
                print(f"\n>> ✗ API 返回错误: {result.get('error', '未知错误')}")
                return False
        else:
            print(f"\n>> ✗ HTTP 错误: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   错误信息: {error_data.get('error', response.text)}")
            except:
                print(f"   响应内容: {response.text[:500]}")
            return False
            
    except requests.exceptions.Timeout:
        print("\n>> ✗ 请求超时，故事可能太长，请尝试分段处理")
        return False
    except requests.exceptions.ConnectionError:
        print(f"\n>> ✗ 无法连接到 API 服务器: {api_url}")
        print("   请确保服务器正在运行")
        return False
    except Exception as ex:
        print(f"\n>> ✗ 发生错误: {str(ex)}")
        import traceback
        traceback.print_exc()
        return False


def generate_srt(subtitles: list, output_path: str):
    """
    生成 SRT 格式字幕文件
    
    Args:
        subtitles: 字幕数据列表
        output_path: 输出文件路径
    """
    def format_time(seconds: float) -> str:
        """将秒数转换为 SRT 时间格式 HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, subtitle in enumerate(subtitles, 1):
            f.write(f"{idx}\n")
            start_time = format_time(subtitle['start'] / 1000)  # 转换毫秒为秒
            end_time = format_time(subtitle['end'] / 1000)      # 转换毫秒为秒
            f.write(f"{start_time} --> {end_time}\n")
            
            # 添加说话人信息（如果有）
            speaker = subtitle.get('speaker', '')
            text = subtitle['text']
            if speaker and speaker != "旁白":
                f.write(f"[{speaker}] {text}\n")
            else:
                f.write(f"{text}\n")
            f.write("\n")


def generate_story_audio_single_api(
    api_url: str,
    story_item: dict,
    output_dir: str = "outputs",
    output_index: int = 1
):
    """
    调用单个故事条目 API 生成音频和字幕
    
    Args:
        api_url: API 服务器地址
        story_item: 单个故事数据对象
        output_dir: 输出目录
        output_index: 输出文件编号（用于命名）
    
    Returns:
        是否成功
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 准备请求数据
    request_body = {
        "story_data": story_item
    }
    
    # 调用 API
    print(f">> 调用 API: {api_url}/generate_story_audio_single")
    print(f">> 处理条目 {output_index}: {story_item.get('text', '')[:50]}...")
    
    try:
        response = requests.post(
            f"{api_url}/generate_story_audio_single",
            json=request_body,
            headers={"Content-Type": "application/json"},
            timeout=600  # 10分钟超时
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("status") == "success":
                # 保存音频文件
                audio_base64 = result['audio']
                audio_bytes = base64.b64decode(audio_base64)
                
                # 生成输出文件名
                audio_output_path = Path(output_dir) / f"{output_index}.wav"
                subtitle_output_path = Path(output_dir) / f"{output_index}_subtitles.json"
                
                # 保存音频
                with open(audio_output_path, 'wb') as f:
                    f.write(audio_bytes)
                print(f"   ✓ 音频已保存: {audio_output_path}")
                
                # 保存字幕（JSON 格式）
                subtitles = result['subtitles']
                with open(subtitle_output_path, 'w', encoding='utf-8') as f:
                    json.dump(subtitles, f, ensure_ascii=False, indent=2)
                print(f"   ✓ 字幕已保存: {subtitle_output_path}")
                
                return True
            else:
                print(f"   ✗ API 返回错误: {result.get('error', '未知错误')}")
                return False
        else:
            print(f"   ✗ HTTP 错误: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   错误信息: {error_data.get('error', response.text)}")
            except:
                print(f"   响应内容: {response.text[:500]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"   ✗ 请求超时")
        return False
    except requests.exceptions.ConnectionError:
        print(f"   ✗ 无法连接到 API 服务器: {api_url}")
        return False
    except Exception as ex:
        print(f"   ✗ 发生错误: {str(ex)}")
        import traceback
        traceback.print_exc()
        return False


def generate_story_audio_single_batch(
    api_url: str,
    json_file_path: str,
    output_dir: str = "outputs",
    item_count: int = 5,
    case_name: str = "single_batch"
):
    """
    批量调用单个故事条目 API，依次处理前N个条目
    
    Args:
        api_url: API 服务器地址
        json_file_path: 故事 JSON 文件路径
        output_dir: 输出目录
        item_count: 要处理的条目数量（前N个）
        case_name: 测试用例名称（用于输出文件命名）
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 读取 JSON 文件
    print(f">> 读取 JSON 文件: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        story_data = json.load(f)
    
    if not isinstance(story_data, list):
        print(f"✗ 错误: JSON 文件必须是数组格式")
        return False
    
    # 获取前N个条目
    items_to_process = story_data[:item_count]
    total_items = len(story_data)
    
    print(f">> 文件中共有 {total_items} 个条目，将处理前 {len(items_to_process)} 个")
    print(f">> 输出目录: {output_dir}")
    print(f">> 开始批量处理...\n")
    
    # 记录开始时间
    start_time = time.time()
    
    # 依次处理每个条目
    success_count = 0
    failed_count = 0
    
    for idx, item in enumerate(items_to_process, 1):
        print(f"\n[{idx}/{len(items_to_process)}] 处理条目 {idx}")
        success = generate_story_audio_single_api(
            api_url=api_url,
            story_item=item,
            output_dir=output_dir,
            output_index=idx
        )
        
        if success:
            success_count += 1
        else:
            failed_count += 1
    
    # 记录结束时间
    elapsed_time = time.time() - start_time
    
    # 显示总结
    print(f"\n" + "=" * 80)
    print(f"批量处理完成")
    print("=" * 80)
    print(f"   - 总条目数: {len(items_to_process)}")
    print(f"   - 成功: {success_count} 个 ✓")
    print(f"   - 失败: {failed_count} 个 ✗")
    print(f"   - 执行时间: {elapsed_time:.2f} 秒")
    print(f"   - 平均每个条目: {elapsed_time/len(items_to_process):.2f} 秒")
    print(f"\n>> 所有文件已保存到: {output_dir}/")
    
    return failed_count == 0


def run_test_case(test_case: dict, api_url: str, output_dir: str) -> bool:
    """
    运行单个测试用例
    
    Args:
        test_case: 测试用例配置
        api_url: API 服务器地址
        output_dir: 输出目录
        
    Returns:
        测试是否成功
    """
    print("\n" + "=" * 80)
    print(f"测试用例: {test_case['name']}")
    print(f"说明: {test_case['description']}")
    print("=" * 80)
    
    # 检查文件是否存在
    json_file = test_case['json_file']
    voice_map = test_case.get('voice_map')
    use_json_api = test_case.get('use_json_api', False)
    use_single_api = test_case.get('use_single_api', False)
    
    if not Path(json_file).exists():
        print(f"✗ 错误: JSON 文件不存在: {json_file}")
        return False
    
    if voice_map and not Path(voice_map).exists():
        print(f"✗ 错误: 音色映射文件不存在: {voice_map}")
        return False
    
    # 显示配置信息
    print(f"\n配置信息:")
    if use_single_api:
        api_type = "单个条目批量调用"
        item_count = test_case.get('item_count', 5)
        print(f"  - API 类型: {api_type}")
        print(f"  - 处理条目数: {item_count}")
    else:
        api_type = 'JSON 直接调用' if use_json_api else '文件上传'
        print(f"  - API 类型: {api_type}")
    print(f"  - JSON 文件: {json_file}")
    print(f"  - 音色映射: {voice_map if voice_map else '未使用（使用默认映射）'}")
    print(f"  - 输出目录: {output_dir}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 根据配置选择调用方式
    if use_single_api:
        # 使用单个条目批量调用方式
        item_count = test_case.get('item_count', 5)
        success = generate_story_audio_single_batch(
            api_url=api_url,
            json_file_path=json_file,
            output_dir=output_dir,
            item_count=item_count,
            case_name=test_case['output_suffix']
        )
    elif use_json_api:
        # 使用 JSON 直接调用方式
        success = generate_story_audio_json_api(
            api_url=api_url,
            json_file_path=json_file,
            output_dir=output_dir,
            voice_map_path=voice_map,
            case_name=test_case['output_suffix']
        )
    else:
        # 使用文件上传方式
        success = generate_story_audio(
            api_url=api_url,
            json_file_path=json_file,
            output_dir=output_dir,
            voice_map_path=voice_map,
            case_name=test_case['output_suffix']
        )
    
    # 记录结束时间
    elapsed_time = time.time() - start_time
    
    # 显示结果
    print(f"\n测试结果: {'✓ 通过' if success else '✗ 失败'}")
    print(f"执行时间: {elapsed_time:.2f} 秒")
    
    return success


def run_all_tests(api_url: str = BASE_URL, output_dir: str = "outputs"):
    """
    运行所有测试用例
    
    Args:
        api_url: API 服务器地址
        output_dir: 输出目录
    """
    print("\n" + "🚀" * 40)
    print("故事音频生成 API 自动测试工具")
    print("🚀" * 40)
    print(f"\nAPI 地址: {api_url}")
    print(f"输出目录: {output_dir}")
    print(f"测试用例数: {len(TEST_CASES)}")
    
    # 运行所有测试用例
    results = []
    for idx, test_case in enumerate(TEST_CASES, 1):
        print(f"\n[{idx}/{len(TEST_CASES)}] 开始执行测试用例...")
        success = run_test_case(test_case, api_url, output_dir)
        results.append({
            'name': test_case['name'],
            'success': success
        })
    
    # 显示总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    passed = sum(1 for r in results if r['success'])
    failed = len(results) - passed
    
    for idx, result in enumerate(results, 1):
        status = "✓ 通过" if result['success'] else "✗ 失败"
        print(f"{idx}. {result['name']}: {status}")
    
    print(f"\n总计: {len(results)} 个测试")
    print(f"通过: {passed} 个 ✓")
    print(f"失败: {failed} 个 ✗")
    print(f"成功率: {(passed/len(results)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n⚠️  有 {failed} 个测试失败，请检查错误信息")
    
    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="故事音频生成 API 自动测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行所有测试用例（使用默认配置）
  python api_story_client.py
  
  # 指定 API 地址
  python api_story_client.py --api-url http://192.168.1.100:6006
  
  # 指定输出目录
  python api_story_client.py --output-dir test_results
  
  # 运行单个测试用例
  python api_story_client.py --case 1
        """
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=BASE_URL,
        help=f"API 服务器地址 (默认: {BASE_URL})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="输出目录 (默认: outputs)"
    )
    parser.add_argument(
        "--case",
        type=int,
        default=None,
        help="运行指定的测试用例编号（1-N），不指定则运行所有用例"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用的测试用例"
    )
    
    args = parser.parse_args()
    
    # 列出测试用例
    if args.list:
        print("\n可用的测试用例:")
        for idx, test_case in enumerate(TEST_CASES, 1):
            print(f"\n{idx}. {test_case['name']}")
            print(f"   描述: {test_case['description']}")
            if test_case.get('use_single_api', False):
                api_type = "单个条目批量调用"
                item_count = test_case.get('item_count', 5)
                print(f"   API 类型: {api_type}")
                print(f"   处理条目数: {item_count}")
            else:
                api_type = 'JSON 直接调用' if test_case.get('use_json_api', False) else '文件上传'
                print(f"   API 类型: {api_type}")
            print(f"   JSON: {test_case['json_file']}")
            print(f"   音色映射: {test_case.get('voice_map', '无')}")
        return
    
    # 运行指定的测试用例
    if args.case is not None:
        if 1 <= args.case <= len(TEST_CASES):
            test_case = TEST_CASES[args.case - 1]
            print("\n" + "🚀" * 40)
            print("故事音频生成 API 单用例测试")
            print("🚀" * 40)
            success = run_test_case(test_case, args.api_url, args.output_dir)
            if success:
                print("\n✓ 测试完成!")
            else:
                print("\n✗ 测试失败，请查看错误信息")
        else:
            print(f"错误: 测试用例编号必须在 1-{len(TEST_CASES)} 之间")
        return
    
    # 运行所有测试用例
    success = run_all_tests(args.api_url, args.output_dir)
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()

