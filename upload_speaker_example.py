#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说话人音频上传示例

展示如何使用 /upload_speaker API 上传音频文件，然后在 TTS 生成中使用
"""

import requests
import base64
import os
import time

USE_LOCAL = False
API_BASE_URL = "http://localhost:6006" if USE_LOCAL else "http://117.50.176.241:6006"

def upload_speaker_audio(audio_file_path: str):
    """
    上传说话人音频文件
    
    参数:
        audio_file_path: 本地音频文件路径
    
    返回:
        dict: 包含上传结果的字典，其中 file_path 可用于后续 TTS 调用
    """
    if not os.path.exists(audio_file_path):
        print(f"错误: 文件不存在 - {audio_file_path}")
        return None
    
    print(f">> 上传音频文件: {audio_file_path}")
    
    # 读取文件
    with open(audio_file_path, 'rb') as f:
        files = {'file': (os.path.basename(audio_file_path), f, 'audio/wav')}
        
        # 发送上传请求
        response = requests.post(f"{API_BASE_URL}/upload_speaker", files=files)
    
    if response.status_code == 200:
        result = response.json()
        if result.get('status') == 'success':
            print(f"✓ 上传成功!")
            print(f"  文件路径: {result['file_path']}")
            print(f"  文件名: {result['filename']}")
            print(f"  文件大小: {result['file_size']} 字节")
            return result
        else:
            print(f"✗ 上传失败: {result.get('error', '未知错误')}")
            return None
    else:
        print(f"✗ 请求失败: HTTP {response.status_code}")
        try:
            error_info = response.json()
            print(f"  错误信息: {error_info.get('error', '未知错误')}")
        except:
            print(f"  响应内容: {response.text}")
        return None


def generate_tts_with_uploaded_audio(spk_audio_path: str, text: str, return_subtitle: bool = False):
    """
    使用上传的音频文件进行 TTS 生成
    
    参数:
        spk_audio_path: 上传后返回的文件路径
        text: 要合成的文本
        return_subtitle: 是否返回字幕
    
    返回:
        dict 或 bytes: TTS 生成结果
    """
    print(f"\n>> 生成语音...")
    print(f"    参考音频: {spk_audio_path}")
    print(f"    文本: {text[:50]}..." if len(text) > 50 else f">> 文本: {text}")

    
    data = {
        "text": text,
        "spk_audio_path": spk_audio_path,
        "return_subtitle": return_subtitle
    }
    
    response = requests.post(f"{API_BASE_URL}/tts_url", json=data)
    
    if response.status_code == 200:
        if return_subtitle:
            result = response.json()
            print(f"✓ 生成成功!")
            print(f"  字幕片段数: {result.get('subtitle_count', 0)}")
            return result
        else:
            audio_bytes = response.content
            print(f"✓ 生成成功!")
            print(f"  音频大小: {len(audio_bytes)} 字节")
            return audio_bytes
    else:
        print(f"✗ 生成失败: HTTP {response.status_code}")
        try:
            error_info = response.json()
            print(f"  错误信息: {error_info.get('error', '未知错误')}")
        except:
            print(f"  响应内容: {response.text}")
        return None


def save_audio_from_base64(audio_base64: str, output_path: str):
    """从 base64 保存音频"""
    audio_bytes = base64.b64decode(audio_base64)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(audio_bytes)
    print(f"  音频已保存到: {output_path}")


# ========== 示例用法 ==========

def example_1_basic_upload_and_tts():
    """示例 1: 基本的上传和 TTS 生成"""
    print("\n" + "=" * 80)
    print("示例 1: 上传音频并生成语音（不带字幕）")
    print("=" * 80)
    
    # 步骤 1: 上传音频文件
    upload_result = upload_speaker_audio("test/shan_tian_fang_voices_nano.mp3")
    
    if not upload_result:
        return
    
    # 步骤 2: 使用上传的文件路径进行 TTS
    uploaded_path = upload_result['file_path']
    audio_bytes = generate_tts_with_uploaded_audio(
        spk_audio_path=uploaded_path,
        text="你好，这是使用上传音频文件生成的语音。",
        return_subtitle=False
    )
    
    # 步骤 3: 保存生成的音频
    if audio_bytes:
        output_path = "outputs/example1_output.wav"
        os.makedirs("outputs", exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(audio_bytes)
        print(f"  输出音频已保存到: {output_path}")


def example_2_upload_with_subtitle():
    """示例 2: 上传音频并生成语音（带字幕）"""
    print("\n" + "=" * 80)
    print("示例 2: 上传音频并生成语音（带字幕）")
    print("=" * 80)
    
    # 步骤 1: 上传音频文件
    upload_result = upload_speaker_audio("tests/fanqie_speaker.mp3")
    
    if not upload_result:
        return
    
    # 步骤 2: 使用上传的文件路径进行 TTS（带字幕）
    uploaded_path = upload_result['file_path']
    result = generate_tts_with_uploaded_audio(
        spk_audio_path=uploaded_path,
        text="""
叶凡大吃一惊，他竟见到了姬紫月，她虽然在甜甜的笑，但脸色苍白，红唇带着丝丝血迹，站在水塘边，在微微地颤抖。
“你怎么来到了这里？”
她一如过去，虽然身受重伤，但还笑得出来，嘴角弯弯，酒窝隐现。
“你难道也被孔雀王追杀了？”叶凡快步向前走去。
“真的遇到了你⋯⋯”她在浅笑，但口中突然溢出一丝丝血迹，脸色更加苍白了，原本灵动的大眼，此刻有些无神，摇晃了一下，一下子栽倒在水塘间。
姬紫月伤势之重，让叶凡心中一惊，其脉搏极其微弱，心跳都快停止了，五脏六腑皆有裂痕，能够活下来，实在是命大。
叶凡运转神力，将其湿漉漉的衣裙蒸干，而后几步就消失在小村外，来到一座山崖上。他取出玉净瓶，向其口中倒入一些神泉，顿时化成一股生命精能，冲进的她的体内。
取自九座圣山的神泉，具有不可思议之灵效，几可生死人肉白骨，甚是不凡，可以感知到，姬紫月伤势瞬间稳住了。
此刻，她紧闭双眸，身体瑟瑟发抖，轻声梦呓，喃喃着什么。叶凡仔细倾听，也难以辨清。
“不要杀我哥哥⋯⋯”她在昏迷中惊叫了一声。
半个时辰之后，姬紫月长长的睫毛轻颤，缓缓睁开了眼睛，想要坐起来，但是却没有成功。
“你的伤势很重，现在虽然稳住了，但不宜妄动。”叶凡将其扶起，让她背靠软藤，坐在山崖上。
“真是可怕⋯⋯”姬紫月似乎有心有余悸。
“你是在说孔雀王吗？”叶凡自然联想到了那位妖族大能。
“是他。”姬紫月点了点头，道：“小毛孩，你怎么一声不响的就溜了，我追了你三千里，结果无功而返。”
“再不走的话，你哥哥会把我分尸，细细研究的。”
姬紫月轻笑，道：“哪里有你说得那么坏，放心好了，只要有我在，你去我们姬家很安全，肯定没有人会为难你。”
叶凡闻言，心中一动，觉得姬紫月很不一般，被姬家颇为看重。
“孔雀王真的太强大了⋯⋯”姬紫月露出忧色，很担心，道：“不知道我皓月哥哥能否逃过这一劫。”
“孔雀王也对你出手了？你是怎么逃出来的，感觉很难逃生。”
孔雀王之强大不可想象，保护姬家兄妹的几位名宿根本无法抵抗，在其面前犹如稻草人一般，不堪一击。
“他真的太恐怖了，法力滔天⋯⋯”到了现在，姬紫月还心有恐惧，他们想开启域门，横渡虚空，但孔雀王一指点出，虚空粉碎，将所有人都震了出来。
当场便有人形神俱灭，姬紫月若不是身怀秘宝，根本无法活下来。
关键时刻，姬家遣出的强者寻到了他们，不然没有一个人可以活下来。
纵然如此，也没有人可以拦下孔雀王，他所向披靡，无人可以阻挡他的步伐，一声大吼，当场让不少人粉碎，化成了血雾，称得上神威震世。
“这也⋯⋯太不可思议了！”叶凡深深震惊不已，这种强大的人物实在是可怕，实力无法揣测。
“堪比上古大能的存在，究竟有多么强大的法力，难以真正说清⋯⋯”
那批强者仅仅拖延了片刻钟，便被打得四分五裂，不过却为姬皓月与姬紫月赢得了时间，让他们远遁而去。
听到这里，叶凡更加确信，姬紫月地位很不一般，纵然在那种时刻，姬家强者都拼死为她创造逃生的机会，不肯舍弃。
“孔雀王与天地合一，吼动河山，声音如惊涛拍岸，一座山峰都被其音震碎了！”
叶凡感觉心惊肉跳，这种威势超乎想象。
“我哥哥的神体，天生具有奇异伟力，可是被彻底压制了，在其面前连凡人都不如。”
堪比上古大能的存在，其无上威势，让人感觉不可思议。
姬皓月有多么强，叶凡曾亲眼见到过，斩杀大妖，犹如拔草摘花，纵横冲击，所向披靡。可是强大如斯，在孔雀王面前，亦微不足道。
“天人合一，掌控一切，几可以摘星捉月⋯⋯”姬紫月到了现在，还有些紧张，秀拳攥的很紧。
“你是怎么逃出来的？”叶凡问道。
“我与哥哥分开逃跑的，他去追我哥哥了⋯⋯”姬紫月有隐忧，那样的大能，根本无法抗衡，纵是姬皓月为神体，修成了三个秘境，真正对上的话，也根本没有任何悬念。
“还没有坏消息传来，想必你哥哥不会那么容易死去。”说到这里，叶凡露出狐疑的神色，道：“你怎么没有回姬家，反倒来到了这里？”
“孔雀王没有理会我，但是有其他人追杀我⋯⋯”这一路上，姬紫月被人不断追杀，身受重创。
急匆匆，她自一个小门派的域门，横渡虚空，来到了此地。她不会刻印道纹，根本不能确定目的地。
“还真是巧⋯⋯”叶凡自语。
“不是碰巧。”姬紫月白了他一眼，道：“我在你身上留有印记，出现在这片地域后，慢慢寻到了这里。”
“什么，你竟在我身上做手脚？”
“我又没有恶意，你紧张什么？当日见你那样反常，就知道你要离开，故此有所准备，留下了印记。结果你跑得太快了，我追不上，距离太远，无法感应到。”
叶凡仔细内视，却没有发现什么异常，道：“你到底做了什么手脚？”
“你慢慢找吧⋯⋯”姬紫月轻笑了起来，但很快又露出了忧色，道：“皓月哥哥，恐怕真的难以逃过这一劫。”
突然，叶凡神情一怔，就在不远处的天空上，出现一道门户，有数名修士冲了出来。
“呀，是追杀我的人！”姬紫月轻声叫了出来，道：“他们开启域门，横渡虚空，跟了下来。”
“哧！”
一道可怖的闪电劈落下来，那些人非常果断，发现目标，立刻展出神通，向下轰杀。
叶凡扶起姬紫月，躲向一旁，银色的闪电一下子将山崖击穿，乱石四射，崩飞向各处。
“轰！”
天空中，一座古朴的塔快速放大，向下镇压而来。
叶凡当场就变了颜色，使用塔、钟、鼎的人，一般都不是凡俗之辈，多半都强大无比。
天空中，那座塔不过五层，但却像是大山一般沉重，压的人喘不过气来，弥漫出让人心悸的气息。
叶凡拉起姬紫月，快速离开山崖，冲向远方，这些人太强大了，力敌的话肯定会吃大亏。
老疯子传给叶凡的步法，乃是一种无上秘术，具有不可思议之伟力，速度极快，时间不长，已经在十里之外。
他没有停留，继续远遁，直至到了百里外，才长出一口气，道：“那几人很不一般⋯⋯”
“都非常强大，不然我也不会逃了。”姬紫月的小虎牙亮晶晶，轻轻磨动。
“看起来不像是妖族人。”叶凡露出疑惑的神色。
姬紫月咬着红唇，道：“有人想借机除掉我与皓月哥哥。”
“该不会又是那个姬碧月吧？”
“不是，比她惹出的麻烦大多了，肯定是有人不愿我们家族有神体成长起来。纵然皓月哥哥能够逃出孔雀王的手心，也很危险，恐怕会有人族强者暗中出手，趁此次机会将他击杀，栽在妖族的头上。”
姬紫月虽然年岁不大，但是却想的很多，预料到了可能会发生的一些不好的事情。
“他们应该盯住姬皓月才对，为什么也要对你不利，难道你也是神体？”
“我怎么可能是神体⋯⋯”姬紫月没有多说，略微思索了片刻，而后瞥了一眼叶凡，调笑道：“小屁孩，你身上的秘密真的很多，你的那块绿铜到底是什么，你的肉体为什么可以挡住神体的攻击，你怎么学会了疯老人的步法，你为什么要急着逃离太玄？”
这个小丫头心思很活，古灵精怪，明显是想转移话题，反倒探寻起了叶凡的秘密。
突然，一道金光横空而过，撕裂出数十道强大的电芒，如一条条金蛇在舞动。
“是闪电鸟。”姬紫月蹙眉，突然叫道：“不好，是凤冠闪电鸟，非常强大的一种异禽，我们避开，不要闯入它的领地。”
远去的那只闪电鸟通体金黄，像是黄金浇铸而成，电芒炽烈，其头顶上微微隆起，犹如凤冠，看起来非常不凡。
这是一种强大的异禽，根本不是寻常的闪电鸟所能够比拟的，拥有极其恐怖的攻击力，绝大多数修士见到都要退避三舍。
“幸好没有发现我们。”姬紫月暗自庆幸。
“前方多半有凤冠闪电鸟的巢穴，也许可以利用它阻挡追杀你的人。”
“对，有道理。”姬紫月的大眼顿时弯成了月牙状。
两人一路前行，果然在山脉深处发现一个鸟巢，建在一株参天古木上，四枚金光闪闪的鸟蛋足有人头那么大，缭绕着一道道金色的电芒。
“这种异禽非常稀少，可以不断进化，最是难得。真想抱走一枚卵，孵化出来，当作宠物养。”姬紫月有些遗憾地说道。
她未敢这样做，与叶凡一起在大山中留下足够的蛛丝马迹，而后冲天而去。
不久后，那几名强大的修士果然寻到了山脉深处，发现了很多“线索”，在这片地域仔细寻找，觉得姬紫月就藏身在附近。
直至过去很久，几人寻到一株参天古树前。
“鸟巢⋯⋯不好！”当中一人惊叫，道：“这似乎是凤冠闪电鸟的巢穴，这种异禽万万招惹不得，若是被其误会我等要取卵，肯定会不死不休。”
巨大的鸟巢中，里面有四枚鸟蛋，蛋壳仿若黄金铸成，绚烂夺目。
“快走，这种强大的异禽最是记仇，如果被堵在这里，后果不堪设想！”
可是，他们在这里转悠了很长时间，外出的凤冠闪电鸟已经回返，发出摄人心魄的鸣啸，在天空中留下一道长长的金色残影，俯冲了下来。
叶凡与姬紫月在远山眺望，只见那片山脉深处闪电冲天，他们没有停留，快速远离而去。
第二日，叶凡在一片秀丽的山峦间，见到一个十六七岁的少年，身材有些单薄，显得有些柔弱，容貌很清秀，眼神清亮，如湖水一般澄净，乌发很柔软，根根轻灵。
这个少年如深山中的一道清泉，给人一股清新的感觉，很有自然的味道。
在这人迹罕至的山脉中，见到这样一个少年，叶凡自然有些惊讶，他带着笑容问道：“小兄弟，能否告知这是何处地界？”
那个少年静静地站在前方，柔软的乌发轻轻飘动，眸子清澈无比，答道：“栖霞山脉，地处晋国界内。”
“小兄弟，你相貌清雅，灵动自然，想来一定是修士吧？”叶凡笑着问道。
“是的。”少年点了点头。
后方，姬紫月脸色雪白，声音颤抖，道：“小毛孩⋯⋯快回来！”
叶凡回头，见她满脸惶恐，大眼中充满了惧意，立时感觉不妙，轻飘飘地后退。
“怎么了，他是谁？”
“他是⋯⋯孔雀王！”姬紫月声音发颤，一扫往昔的活泼开朗，此刻只剩下了惊惧。
“什么？！”叶凡大吃一惊，眼前这个少年是孔雀王？这实在让人感觉突兀与震惊。
“孔雀王你是大能，屹立在绝巅，俯视天下，不要乱杀无辜，放走他吧，与他无关。”姬紫月上前，请求放叶凡离去。
妖族大能孔雀王威慑南域，八百年前便已所向披靡，昔日的摇光圣主都奈何不得他，很少有人知晓他到底活了多么久的岁月。
据外界猜测，他化成人形后，最保守的估计，也有两千多岁了，如果与星空的彼岸对比，那便是秦汉时期的人。
可是，眼下这个少年，看起来不过十六七岁的样子，眸子清澈如水，没有一点沧桑的感觉，实在让人难以联想到孔雀王身上。
姬紫月曾经讲述过他的无上威势，与天地合一，吼动山河，声音如惊涛拍岸，一座山峰都被其音震碎了。
在叶凡想来，孔雀王一定伟岸如山，气势狂霸无匹。可是，万万没有想到，竟是如此的清秀，如深山的一股清泉，似冰峰的一株雪莲，若净土的一道清风。
想到刚才与孔雀王称兄道弟，甚至想过去拍拍他的肩头，叶凡顿时一阵头皮发麻，这可是一位法力滔天，堪比上古大能的存在啊。   
        """,
        return_subtitle=True
    )
    
    if result:
        # 步骤 3: 保存音频
        save_audio_from_base64(result['audio'], "outputs/example3_output.wav")
        
        # 步骤 4: 打印字幕
        print(f"\n  字幕信息:")
        for i, subtitle in enumerate(result['subtitles'], 1):
            duration = subtitle['time_end'] - subtitle['time_begin']
            print(f"    [{i}] {subtitle['text']}")
            print(f"        时间: {subtitle['time_begin']:.0f}ms - {subtitle['time_end']:.0f}ms ({duration/1000:.2f}s)")


def example_3_batch_upload():
    """示例 3: 批量上传多个说话人音频"""
    print("\n" + "=" * 80)
    print("示例 3: 批量上传多个说话人音频")
    print("=" * 80)
    
    audio_files = [
        "examples/voice_02.wav",
        "examples/voice_03.wav",
        "examples/voice_04.wav"
    ]
    
    uploaded_speakers = []
    
    for audio_file in audio_files:
        if os.path.exists(audio_file):
            print(f"\n>> 上传: {audio_file}")
            result = upload_speaker_audio(audio_file)
            if result:
                uploaded_speakers.append({
                    'original': audio_file,
                    'uploaded_path': result['file_path'],
                    'filename': result['filename']
                })
    
    print(f"\n>> 成功上传 {len(uploaded_speakers)} 个音频文件:")
    for speaker in uploaded_speakers:
        print(f"  - {speaker['original']} -> {speaker['uploaded_path']}")
    
    # 使用第一个上传的音频生成语音
    if uploaded_speakers:
        print(f"\n>> 使用第一个上传的音频生成语音...")
        audio_bytes = generate_tts_with_uploaded_audio(
            spk_audio_path=uploaded_speakers[0]['uploaded_path'],
            text="这是使用批量上传的第一个音频生成的语音。",
            return_subtitle=False
        )
        
        if audio_bytes:
            output_path = "outputs/example3_output.wav"
            os.makedirs("outputs", exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(audio_bytes)
            print(f"  输出音频已保存到: {output_path}")


def example_4_error_handling():
    """示例 4: 错误处理"""
    print("\n" + "=" * 80)
    print("示例 4: 错误处理示例")
    print("=" * 80)
    
    # 测试 1: 上传不存在的文件
    print("\n>> 测试 1: 上传不存在的文件")
    upload_speaker_audio("non_existent_file.wav")
    
    # 测试 2: 上传不支持的格式（如果有的话）
    print("\n>> 测试 2: 尝试上传文本文件")
    # 创建一个临时文本文件
    temp_file = "temp_test.txt"
    with open(temp_file, 'w') as f:
        f.write("This is not an audio file")
    upload_speaker_audio(temp_file)
    os.remove(temp_file)  # 清理
    
    # 测试 3: 使用不存在的路径进行 TTS
    print("\n>> 测试 3: 使用不存在的音频路径进行 TTS")
    generate_tts_with_uploaded_audio(
        spk_audio_path="uploads/speakers/non_existent.wav",
        text="这应该会失败",
        return_subtitle=False
    )


def example_5_complete_workflow():
    """示例 5: 完整工作流程"""
    print("\n" + "=" * 80)
    print("示例 5: 完整工作流程（上传 -> 生成 -> 保存字幕）")
    print("=" * 80)
    
    # 步骤 1: 上传
    upload_result = upload_speaker_audio("examples/voice_02.wav")
    if not upload_result:
        return
    
    # 步骤 2: 生成（带字幕）
    result = generate_tts_with_uploaded_audio(
        spk_audio_path=upload_result['file_path'],
        text="离开神城已经一个多月了，重新回来，叶凡感受到了一种亲切。这是一个充满激情的巨城。",
        return_subtitle=True
    )
    
    if not result:
        return
    
    # 步骤 3: 保存音频
    save_audio_from_base64(result['audio'], "outputs/example5_output.wav")
    
    # 步骤 4: 保存字幕为 JSON
    import json
    subtitle_path = "outputs/example5_subtitle.json"
    with open(subtitle_path, 'w', encoding='utf-8') as f:
        json.dump(result['subtitles'], f, ensure_ascii=False, indent=2)
    print(f"  字幕已保存到: {subtitle_path}")
    
    # 步骤 5: 保存字幕为 SRT
    def format_srt_time(ms):
        s = int(ms / 1000)
        ms = int(ms % 1000)
        h, m = s // 3600, (s % 3600) // 60
        s = s % 60
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    
    srt_path = "outputs/example5_subtitle.srt"
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, sub in enumerate(result['subtitles'], 1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(sub['time_begin'])} --> {format_srt_time(sub['time_end'])}\n")
            f.write(f"{sub['text']}\n\n")
    print(f"  SRT 字幕已保存到: {srt_path}")


if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs("outputs", exist_ok=True)
    
    try:
        # example_1_basic_upload_and_tts()
        start_time = time.time()
        example_2_upload_with_subtitle()
        end_time = time.time()
        print(f"耗时: {end_time - start_time:.2f} 秒")
        # example_3_batch_upload()
        # example_4_error_handling()
        # example_5_complete_workflow()
        
        print("\n" + "=" * 80)
        print("所有示例执行完成！")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("\n错误: 无法连接到 API 服务器")
        print("请确保服务器正在运行: python api_server.py")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

