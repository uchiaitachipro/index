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
    upload_result = upload_speaker_audio("test/shan_tian_fang_voices_nano.mp3")
    
    if not upload_result:
        return
    
    # 步骤 2: 使用上传的文件路径进行 TTS（带字幕）
    uploaded_path = upload_result['file_path']
    result = generate_tts_with_uploaded_audio(
        spk_audio_path=uploaded_path,
        text="""
火域外，足有上百名骑士守护姬紫月，叶凡料想华云飞如此心机肯定不会贸然动手而留下破绽。
姬紫月静静立身孤坟前，若天阙仙子误坠凡尘，如薄云掩明月，似轻风拂玉花，集灵秀于一身。
轻风拂过，第一层火域内赤焰腾腾。
此刻，叶凡眉心那汪金色的小湖，深邃如渊，神识化形而出，凝聚成一道细丝，道：“昔日追杀你的麻衣人听命于华云飞。”
以神识传音完毕，叶凡冲向火域深处，他相信姬紫月若是有提防，对方很难有可乘之机。
他径直来到第七层火域，祭出自己的鼎，收取五彩云焰，恐怖的火能如小溪、若河流，不断淌进鼎内。
即将离开，他想准备一些保命的手段，不淬炼鼎，只是纳于当中，纵收于轮海间，也不会给他带来伤害。
最终，五色云焰将鼎彻底充满，再也没有一丝空间可以利用，万物母气封鼎，叶凡将之纳于体内。
当远离火域后，叶凡深深地望了一眼，而后冲向北方。
数日后，叶凡得知了一则消息，让他心中惊喜。
如今，北域风波不断，皆因“源”而起，南域很多修士都想去碰碰运气。
摇光圣地与姬家，早在半年前就已经分批遣出强者，前往北域。他们屹立世间，深不可测，域门打开，可以直接横渡到北域。
可是对于散修来说就比较麻烦了，想去北域最起码要飞上数年。为此，大多数人只能另想他法，比如，付出不菲的代价，借道前往北域。
半个月后，叶凡来到了逍遥门，传说此地为两条龙脉汇聚而成，山峰林立，像是龙骨，连绵起伏，多奇树异草，巍峨而不失秀丽。
在这片地域，它是仅次于圣地的超级大势力，与太玄门不相上下，如今正步入鼎盛时期，门中高手如云。
叶凡来此是为了“偷渡”，按照他所得悉的情况，只要付出不菲的代价，就可借助他们的域门横渡虚空。
山门外，足有上千人在等候。
“兄弟你年龄这么小，也想去北域碰运气？”一个满脸络腮胡须的大汉问道。
“是啊，想去搏上一搏。”叶凡答道。
“路途遥远，北域凶险，年纪这么小，最好不要冒险。”一个枯瘦如柴的老人这样劝道。
这千余人有男有女，修为很难说清，参差不齐。
山门大开，走出几名修士，为首的是一个中年人，脸色蜡黄，步履飘忽。
“我说诸位，你们真是运气，这可能是最后一次了。因为，掌教出关后，得悉这一切，很不高兴。”
人群顿时一阵议论，很多人都很庆幸。
众人陆续上前，无需黄脸人多说什么，纷纷呈现宝物，想要横渡虚空，不花费巨大代价是不可能的。
“命泉修士的武器，这也拿得出手？”黄脸中年人轻蔑地扫了一眼，直接将第一个修士拒绝。
“请前辈通融。”那名修士再三恳求，道：“这是我祭炼了十几年的武器，已经是身上最珍贵的器物了。”
“你就是祭炼过一百年也无用，不要说是你，就是彼岸境界的修士所锤炼的武器，若是寻常之物，我们也不会收。”黄脸中年人露出鄙夷之色。
这名修士非常不甘，但却不敢再多说什么，怏怏退去。
第二名修士呈上一面铜镜，古意盎然，不过有些缺陷，上面有三道裂纹。
“我们不收这种次品！”
“这可是道宫巅峰修士祭炼出的武器，威力强绝。”那名修士小心的解释。
“再好的武器，出现致命的裂痕，也已是废品，有何用处，纵是温养上百年，也无法恢复。”黄脸中年人露出不耐烦的神色，挥了挥手，让他离开。
“赤月果一枚！”第三名修士上前，呈上一个玉盒，打开后露出一个红彤彤的果实，状若弯月，色泽晶莹，香气扑鼻。
黄脸中年人双目中露出一丝贪色，“啪”的一声合上玉盒，道：“罕见的灵果，可以通过。”
“紫耀铜精一块！”
当连续数人未能通过后，第九人呈上一块紫铜，有点点光彩闪耀。
“真的是紫耀铜精，修行到第三秘境的名宿，最需这样的炼器材料，可以通过。”黄脸中年人眼中火热，恋恋不舍的将铜精交给身边的人，让那名修士通过。
想要在此借道，必须要花费极大的代价，有大半人提供的宝物与灵药都不被接受。
“又不能直达北域，竟还要出这样的天价！”不能通过的人很不满，在旁议论纷纷。
除却圣地外，没有任何一个门派可以直接横渡到北域，纵然如此，还是有许多散修来此借道。
轮到叶凡时，他掏出一块羊脂白玉般的神铁，不过龙眼大，顿时将黄脸中年人惊得跳了起来，眼中满是贪婪之色，颤声道：“这⋯⋯难道是⋯⋯传说中的神铁？”
小铁块洁白如玉，吸引了很多人的目光，黄脸中年人捧在手中，反复观看，最终叹了一口气，道：“可惜，被炼废了，神精已散，真是暴殄天物！”
“前辈，这块铁价值够吗？”叶凡问道。
此乃乌鸦道人炼废的神铁，他虽然没有进入第八层火域，但却祭出自己的鼎，费了九牛二虎之力，将那些雪白的铁块弄了出来。
仔细观察后，他觉得确实已废，内蕴的精华几近干涸，不再是绝世稀珍。
“你还有这样的神铁吗？”黄脸中年人询问叶凡。
叶凡摇头，道：“我怎么可能拥有这样的神铁，这还是偶然间在一座古洞中得到的。”
黄脸中年人深感可惜，一张蜡黄的脸阴晴不定，道：“虽然废了，但也是珍品，名宿炼器，若是加入当中，可以提升品质，算你通过。”
半个时辰之后，彻底结束，千余名散修只有四百多人留下，皆负付出了天价。
“三日后，你们来此，过时不候。”黄连中年人让他们留下印记，说完这句话起身走向山门。
横渡虚空，并不是一件轻松的事情，必须要仔细刻印道纹，不能有一点瑕疵，不然很有可能会出现意外。这么多散修来此借道，逍遥门不敢掉以轻心，每次都要花费数日时间来准备。
“黄兄别来无恙。”天空中降下几道人影。
那个黄脸中年人姓黄，与他的脸色倒是很配，闻言转身，露出笑容，道：“原来是姬家的俊杰，你们又来查看吗？”
“我们哪里是俊杰，不过是跑腿的而已。”当中一个青年上前道：“奉命行事，不得不如此。”
黄脸中年人满面笑意，非常客气，道：“这次要横渡虚空的人都在此，绝对没有大妖，不信你们自己来看。”
这几个月以来，姬家横扫南域，怒火冲天，追杀不到孔雀王与乌鸦道人，已经连斩了数十名大妖。
“黄兄不要误会，我们也只是随便转转。”纵然是姬家也不愿得罪逍遥门这样的大势力。
姬家的几名年轻人，自这数百人间一一扫过，没有发现妖气，拱了拱手离去。
叶凡都已做好了战斗的准备，见状心中松了一口气。
随后，众人散去，叶凡走的并不快，他避免与姬家的几人相遇。
“小友留步。”刚刚离开十几里，后方就传来传音，只见黄脸中年人一脸的假笑，追了上来。
“前辈有事吗？”叶凡不动声色地问道。
“小友真的没有那种如羊脂玉般的铁块了吗？”黄脸中年人盯着他。
“我只是在古洞中捡到一块而已。”
“说的也是，凭你的修为根本得不到那样的稀世之物。”黄脸中年人点了点头，道：“你带我去那座古洞看一看。”
叶凡顿时皱眉，道：“那座古洞太遥远，飞上数天也难以到达，晚辈难以从命。”
黄脸中年人的脸沉了下来，道：“不过让你带路而已，难道你不想去北域了吗？”
这就是一个真小人，毫不掩饰的威胁。
“三日后就要横渡虚空了，我若带前辈去，肯定会错过机会。”叶凡平静答道。
“你若带我去，下次还有机会。”
叶凡心中鄙夷，明明只是最后一次机会了，这个家伙还真是无耻。
见叶凡不说话，黄脸中年人脸色阴沉，道：“那种神铁对我很重要，今天你必须带我去。”
“我真的分身乏术。”叶凡拒绝。
“敬酒不吃吃罚酒！”黄脸中年人森然道：“别逼我动手。”
叶凡一直在掩饰自己的神力波动，仅仅显化在命泉境界而已，他感应到对方应该在彼岸巅峰，当下点了点头，道：“好吧，我带路。”说罢，他直接向前飞去。
黄脸中年人收起森然杀机，满意地点了点头，露出虚假的笑意，道：“这就对了，我不会亏待你。”
见左右无人，叶凡又向前飞出去数十里，而后立身在空中。
“为什么停下来？”黄脸中年人面色不善，杀意再次浮现。
“送你上路！”叶凡答道。
虚空大手印残式，浮现而出，黑色的大手遮蔽虚空，铺天盖地而下。
“啪！”
黄脸中年人直接被拍碎在空中，连哼都未能哼出一声。对于这样的小人，叶凡不想与之多纠缠，一巴掌拍死就是。
遥远的天际，数人同时变色，正是不久前离去的那几名姬家的年轻子弟，他们立刻发出惊声。
“虚空大手印的波动！”
“不对，不是正宗的虚空大手印，力道混乱！”
三日后，叶凡重新来到逍遥门，与那数百人共同进入这一大派。
两条龙脉交汇，奇峰并立，气势巍峨，同时不乏灵秀。
数百人被引领进一座山谷中，这里地势开阔，建有一座巨大的祭台，正是开启域门之所。
“诸位，祝你们一路顺风！”逍遥门的一名长老大声说道。
数百人陆续登上高台，即将开启域门，横渡虚空。
叶凡长出了一口气，终于要离开南域，彻底摆脱一切麻烦了。
“慢！”就在这时，逍遥门外突然传来大喝，数十人快速飞来。
域门没有开启，那些人影眨眼及至。
这数十人以老者居多，各个精神矍铄，透发出让人心悸的气息，生命力如汪洋一般旺盛，绝不是普通高手。
旁边，有逍遥门的数名长老相陪，共同飞到此地。
“诸位恕罪。”这些人倒也很客气，并没有盛气凌人，对所有人抱拳，尤其是对逍遥门的长老们更是不断告罪。
叶凡的脸当时就绿了，姬家的人来了，足足有二十几位名老者，其中他更是看到了一个熟人⋯⋯姬惠。
当然，最让叶凡心惊得是，最中央那个老人，如深渊一般空虚，如大海一般深邃，不可揣测。
“这该不会是一位大人物吧？”叶凡感觉大事不妙。
“您是姬长空前辈吗？”逍遥门的一位名宿上前，看着姬家正中的那名老者，露出疑惑的神色。
“正是老朽。”
逍遥门的人露出吃惊得神色，其中一人道：“听闻姬长空前辈，已退隐数十年，不想您再次出世，驾临我逍遥门，我去禀告掌教。”
“无需如此，我等即刻就走。”正中的那名老人再次告罪，开口解释了一番。
姬家发现乌鸦道人的踪迹，不仅姬家圣主亲出，更是足足出动了六名太上长老，想要去诛灭老道士。
姬长空正是当中的一名太上长老，他们在路过此地时得到禀报，有未名的修士在此施展出虚空大手印残式。
这让姬家众人心中升起不好的预感，他们一直在怀疑，叶凡或许将大虚空术等传给了孔雀王等妖族修士。
得知这一消息，他们中途停下，来此一观，时逢逍遥派开启域门，他们一下子联想到了很多，快速赶来。
“我等只看一看，马上就走。”姬家的太上长老，神目如电，亲自在人群中扫视。
叶凡暗暗叫苦，在这逍遥门内如何逃走？四面都有禁制，姬家众人更是拦在山门方向，让他的心彻底凉了。        
        """,
        return_subtitle=True
    )
    
    if result:
        # 步骤 3: 保存音频
        save_audio_from_base64(result['audio'], "outputs/example2_output.wav")
        
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

