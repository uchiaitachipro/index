"""
杂音检测 PyTorch CNN 训练脚本 (原始音频版)

使用原始音频波形/梅尔频谱图训练，让 CNN 自动学习特征
适用于小数据集场景，比手工特征更有效

特点：
1. 使用梅尔频谱图作为输入（2D CNN）
2. 只提取音频尾部（杂音发生位置）
3. 数据增强（时间拉伸、噪声添加等）
4. 自动设备选择 (CUDA > MPS > CPU)
"""
import os
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'

import argparse
import numpy as np
import librosa
import json
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
warnings.filterwarnings('ignore')

# 导入 PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    PYTORCH_AVAILABLE = True
except ImportError:
    print("错误: 需要安装 PyTorch: pip install torch")
    exit(1)

# 导入 tqdm 进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("提示: 安装 tqdm 可显示进度条: pip install tqdm")


# ============== 设备检测 ==============
def get_torch_device():
    """获取最佳 PyTorch 设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_device_name(device):
    """获取设备友好名称"""
    if device.type == 'cuda':
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    elif device.type == 'mps':
        return "MPS (Apple Silicon GPU)"
    return "CPU"


# ============== 音频处理 ==============
class AudioConfig:
    """音频处理配置"""
    SAMPLE_RATE = 22050  # 采样率
    TAIL_DURATION_MS = 300  # 提取尾部长度（毫秒）
    N_MELS = 64  # 梅尔频带数
    N_FFT = 1024  # FFT 窗口大小
    HOP_LENGTH = 256  # 跳跃长度
    FIXED_LENGTH = 32  # 固定时间帧数（用于统一输入尺寸）


def load_audio_tail(audio_path, config=AudioConfig):
    """
    加载音频文件的尾部
    
    Args:
        audio_path: 音频文件路径
        config: 音频配置
    
    Returns:
        numpy array: 音频尾部数据
    """
    try:
        y, sr = librosa.load(str(audio_path), sr=config.SAMPLE_RATE, mono=True)
        
        # 计算尾部采样点数
        tail_samples = int(config.SAMPLE_RATE * config.TAIL_DURATION_MS / 1000)
        
        # 提取尾部
        if len(y) >= tail_samples:
            y_tail = y[-tail_samples:]
        else:
            # 如果音频太短，用零填充
            y_tail = np.zeros(tail_samples)
            y_tail[-len(y):] = y
        
        return y_tail, sr
    except Exception as e:
        print(f"加载音频失败 {audio_path}: {e}")
        return None, None


def audio_to_melspec(y, sr, config=AudioConfig):
    """
    将音频转换为梅尔频谱图
    
    Args:
        y: 音频数据
        sr: 采样率
        config: 音频配置
    
    Returns:
        numpy array: 梅尔频谱图 (n_mels, time_frames)
    """
    # 计算梅尔频谱图
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=config.N_MELS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH
    )
    
    # 转换为 dB 刻度
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 固定长度（填充或截断）
    if mel_spec_db.shape[1] < config.FIXED_LENGTH:
        # 填充
        pad_width = config.FIXED_LENGTH - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=-80)
    elif mel_spec_db.shape[1] > config.FIXED_LENGTH:
        # 截断（保留尾部）
        mel_spec_db = mel_spec_db[:, -config.FIXED_LENGTH:]
    
    # 归一化到 [0, 1]
    mel_spec_db = (mel_spec_db + 80) / 80  # 假设最小值约为 -80 dB
    mel_spec_db = np.clip(mel_spec_db, 0, 1)
    
    return mel_spec_db


def augment_audio(y, sr, augment_type='noise'):
    """
    音频数据增强
    
    Args:
        y: 音频数据
        sr: 采样率
        augment_type: 增强类型
    
    Returns:
        增强后的音频
    """
    if augment_type == 'noise':
        # 添加轻微噪声
        noise = np.random.randn(len(y)) * 0.005 * np.max(np.abs(y))
        return y + noise
    elif augment_type == 'volume':
        # 调整音量
        factor = np.random.uniform(0.8, 1.2)
        return y * factor
    elif augment_type == 'shift':
        # 时间平移
        shift = int(np.random.uniform(-0.1, 0.1) * len(y))
        return np.roll(y, shift)
    elif augment_type == 'stretch':
        # 时间拉伸
        rate = np.random.uniform(0.9, 1.1)
        return librosa.effects.time_stretch(y, rate=rate)
    return y


# ============== 数据集 ==============
class AudioNoiseDataset(Dataset):
    """音频杂音检测数据集"""
    
    def __init__(self, audio_files, labels, config=AudioConfig, augment=False):
        """
        Args:
            audio_files: 音频文件路径列表
            labels: 标签列表 (0=正常, 1=杂音)
            config: 音频配置
            augment: 是否进行数据增强
        """
        self.audio_files = audio_files
        self.labels = labels
        self.config = config
        self.augment = augment
        
        # 预加载所有数据（数据量不大时可以这样做）
        self.data = []
        self.valid_indices = []
        
        print("  预加载音频数据...")
        iterator = tqdm(enumerate(audio_files), total=len(audio_files), desc="  加载") if TQDM_AVAILABLE else enumerate(audio_files)
        
        for idx, audio_path in iterator:
            y, sr = load_audio_tail(audio_path, config)
            if y is not None:
                mel_spec = audio_to_melspec(y, sr, config)
                self.data.append((mel_spec, labels[idx], audio_path, y, sr))
                self.valid_indices.append(idx)
        
        print(f"  成功加载 {len(self.data)} 个音频文件")
    
    def __len__(self):
        if self.augment:
            # 数据增强时，对杂音样本额外生成增强样本
            noise_count = sum(1 for _, label, _, _, _ in self.data if label == 1)
            return len(self.data) + noise_count * 3  # 每个杂音样本额外增强3次
        return len(self.data)
    
    def __getitem__(self, idx):
        if idx < len(self.data):
            mel_spec, label, _, _, _ = self.data[idx]
        else:
            # 增强样本
            # 只对杂音样本进行增强
            noise_indices = [i for i, (_, label, _, _, _) in enumerate(self.data) if label == 1]
            aug_idx = (idx - len(self.data)) % len(noise_indices)
            real_idx = noise_indices[aug_idx]
            
            _, label, _, y, sr = self.data[real_idx]
            
            # 随机选择增强方式
            aug_types = ['noise', 'volume', 'shift']
            aug_type = np.random.choice(aug_types)
            y_aug = augment_audio(y, sr, aug_type)
            mel_spec = audio_to_melspec(y_aug, sr, self.config)
        
        # 添加通道维度 (1, n_mels, time_frames)
        mel_spec = mel_spec[np.newaxis, :, :]
        
        return torch.FloatTensor(mel_spec), torch.LongTensor([label])[0]


# ============== 模型定义 ==============
class MelSpecCNN(nn.Module):
    """
    2D CNN 模型，处理梅尔频谱图输入
    输入: (batch, 1, n_mels, time_frames) = (batch, 1, 64, 32)
    """
    
    def __init__(self):
        super().__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # 池化
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # 全连接层
        self.fc1 = nn.Linear(256 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        
        # 其他
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x: (batch, 1, 64, 32)
        
        # 卷积块1
        x = self.relu(self.bn1(self.conv1(x)))  # (batch, 32, 64, 32)
        x = self.pool(x)  # (batch, 32, 32, 16)
        
        # 卷积块2
        x = self.relu(self.bn2(self.conv2(x)))  # (batch, 64, 32, 16)
        x = self.pool(x)  # (batch, 64, 16, 8)
        
        # 卷积块3
        x = self.relu(self.bn3(self.conv3(x)))  # (batch, 128, 16, 8)
        x = self.pool(x)  # (batch, 128, 8, 4)
        
        # 卷积块4
        x = self.relu(self.bn4(self.conv4(x)))  # (batch, 256, 8, 4)
        x = self.adaptive_pool(x)  # (batch, 256, 2, 2)
        
        # 展平
        x = x.view(x.size(0), -1)  # (batch, 256*2*2)
        
        # 全连接
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x


class ResBlock2D(nn.Module):
    """2D 残差块"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + residual)
        return x


class MelSpecResNet(nn.Module):
    """
    带残差连接的 2D CNN 模型（更强大）
    """
    
    def __init__(self):
        super().__init__()
        
        # 初始卷积
        self.conv_in = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3)
        self.bn_in = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
        # 残差块
        self.res1 = ResBlock2D(32, 64)
        self.res2 = ResBlock2D(64, 128)
        self.res3 = ResBlock2D(128, 256)
        
        # 全局池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # 全连接
        self.fc1 = nn.Linear(256 * 2, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 初始卷积
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.pool(x)
        
        # 残差块
        x = self.res1(x)
        x = self.pool(x)
        x = self.res2(x)
        x = self.pool(x)
        x = self.res3(x)
        
        # 双池化
        avg_pool = self.global_avg_pool(x).view(x.size(0), -1)
        max_pool = self.global_max_pool(x).view(x.size(0), -1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        # 全连接
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


# ============== 训练函数 ==============
def train_model(train_loader, val_loader, model, device, epochs=100, lr=0.001, verbose=True):
    """
    训练模型
    
    Args:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        model: 模型
        device: 设备
        epochs: 训练轮数
        lr: 学习率
        verbose: 是否打印详细信息
    
    Returns:
        训练好的模型
    """
    model = model.to(device)
    
    # 计算类别权重
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy())
    class_counts = np.bincount(train_labels)
    weight = torch.FloatTensor([1.0, class_counts[0] / (class_counts[1] + 1) * 2]).to(device)
    
    if verbose:
        print(f"  类别权重: [正常: 1.0, 杂音: {weight[1].item():.2f}]")
    
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 15
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    if verbose:
        print(f"\n开始训练...")
    
    epoch_iterator = tqdm(range(epochs), desc="  训练") if TQDM_AVAILABLE and verbose else range(epochs)
    
    for epoch in epoch_iterator:
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 更新进度条
        if TQDM_AVAILABLE and verbose:
            epoch_iterator.set_postfix({
                'loss': f'{train_loss:.4f}',
                'acc': f'{train_acc:.2%}',
                'val_acc': f'{val_acc:.2%}',
                'best': f'{best_val_acc:.2%}'
            })
        
        # Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                if verbose:
                    print(f"\n  Early stopping at epoch {epoch + 1}")
                break
        
        # 无 tqdm 时的进度显示
        if not TQDM_AVAILABLE and verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, "
                  f"Acc: {train_acc:.2%}, Val Acc: {val_acc:.2%}")
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    
    if verbose:
        print(f"\n  训练完成! 最佳验证准确率: {best_val_acc:.2%}")
    
    return model, history, best_val_acc


def evaluate_model(model, data_loader, device, file_names=None):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=['无杂音', '有杂音']))
    
    print("\n混淆矩阵:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    
    accuracy = np.mean(all_preds == all_labels)
    return accuracy, all_preds, all_labels


def save_model(model, path='noise_detector_cnn_raw.pth'):
    """保存模型"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }, path)
    print(f"模型已保存到: {path}")


def load_model(path='noise_detector_cnn_raw.pth', device=None):
    """加载模型"""
    if device is None:
        device = get_torch_device()
    
    checkpoint = torch.load(path, map_location=device)
    
    if checkpoint['model_class'] == 'MelSpecResNet':
        model = MelSpecResNet()
    else:
        model = MelSpecCNN()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


# ============== 主程序 ==============
if __name__ == "__main__":
    print("="*60)
    print("杂音检测 CNN 训练 (原始音频版)")
    print("="*60)

    parser = argparse.ArgumentParser(description="杂音检测 CNN 训练 (原始音频版)")
    parser.add_argument("--data-dir", type=str, default="./audio_noise_case", help="数据目录，包含 wav 和 noise.txt")
    parser.add_argument("--test-size", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--epochs", type=int, default=100, help="最大训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--batch-size", type=int, default=0, help="固定 batch_size（>0 生效）；否则自动尝试多组 batch_size")
    parser.add_argument("--batch-sizes", type=str, default="", help="批次大小列表（逗号分隔），如 32,16,8；为空则用默认策略")
    parser.add_argument("--model", type=str, default="both", choices=["cnn", "resnet", "both"], help="训练哪个模型")
    args = parser.parse_args()
    
    # 设备检测
    device = get_torch_device()
    print(f"\n设备: {get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}")
    
    # 数据目录
    data_dir = Path(args.data_dir)
    noise_txt = data_dir / "noise.txt"
    
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        exit(1)
    
    # 读取杂音标签
    noise_files = set()
    if noise_txt.exists():
        with open(noise_txt, 'r') as f:
            noise_files = {line.strip() for line in f if line.strip()}
        print(f"杂音标签数量: {len(noise_files)}")
    
    # 收集所有音频文件
    audio_files = list(data_dir.glob("*.wav"))
    print(f"音频文件数量: {len(audio_files)}")
    
    # 创建标签
    labels = []
    valid_files = []
    for audio_file in audio_files:
        file_name = audio_file.stem
        is_noise = file_name in noise_files
        labels.append(1 if is_noise else 0)
        valid_files.append(audio_file)
    
    labels = np.array(labels)
    print(f"有杂音: {np.sum(labels)}, 无杂音: {np.sum(1-labels)}")
    
    # 划分训练集和验证集
    train_files, val_files, train_labels, val_labels = train_test_split(
        valid_files, labels, test_size=args.test_size, random_state=args.seed, stratify=labels
    )
    
    print(f"\n训练集: {len(train_files)}, 验证集: {len(val_files)}")
    
    # 创建数据集
    print("\n创建训练数据集...")
    train_dataset = AudioNoiseDataset(train_files, train_labels, augment=True)
    print("\n创建验证数据集...")
    val_dataset = AudioNoiseDataset(val_files, val_labels, augment=False)
    
    # batch_size 策略：允许显式指定，或默认尝试更小 batch_size
    if args.batch_size and args.batch_size > 0:
        batch_sizes_to_try = [int(args.batch_size)]
    elif args.batch_sizes.strip():
        batch_sizes_to_try = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    else:
        # 默认：GPU(MPS/CUDA) 先用 32，再尝试更小的 16/8（有时小 batch 更容易泛化）
        batch_sizes_to_try = [32, 16, 8] if device.type in ("cuda", "mps") else [16, 8, 4]

    # 训练多个模型配置
    configs = []
    if args.model in ("cnn", "both"):
        configs.append({'name': 'MelSpecCNN', 'model': MelSpecCNN()})
    if args.model in ("resnet", "both"):
        configs.append({'name': 'MelSpecResNet', 'model': MelSpecResNet()})
    
    best_model = None
    best_acc = 0.0
    best_name = ""
    best_batch_size = None

    for batch_size in batch_sizes_to_try:
        # 数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        print(f"\n批次大小: {batch_size}")
        print(f"训练批次数: {len(train_loader)}")
        print(f"验证批次数: {len(val_loader)}")
    
        for config in configs:
            print(f"\n{'='*60}")
            print(f"训练模型: {config['name']} (batch_size={batch_size})")
            print(f"{'='*60}")

            model, history, val_acc = train_model(
                train_loader, val_loader,
                config['model'], device,
                epochs=args.epochs,
                lr=args.lr,
                verbose=True
            )

            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model
                best_name = config['name']
                best_batch_size = batch_size
    
    print(f"\n{'='*60}")
    print(f"最佳模型: {best_name}, batch_size: {best_batch_size}, 验证准确率: {best_acc:.2%}")
    print(f"{'='*60}")
    
    # 在全部数据上评估
    print("\n在验证集上评估:")
    evaluate_model(best_model, val_loader, device)
    
    # 保存模型
    save_model(best_model)
    
    # 保存配置信息
    config_info = {
        'model_type': best_name,
        'batch_size': int(best_batch_size) if best_batch_size is not None else None,
        'best_val_acc': float(best_acc),
        'audio_config': {
            'sample_rate': AudioConfig.SAMPLE_RATE,
            'tail_duration_ms': AudioConfig.TAIL_DURATION_MS,
            'n_mels': AudioConfig.N_MELS,
            'n_fft': AudioConfig.N_FFT,
            'hop_length': AudioConfig.HOP_LENGTH,
            'fixed_length': AudioConfig.FIXED_LENGTH,
        }
    }
    with open('noise_detector_cnn_raw_info.json', 'w') as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)

