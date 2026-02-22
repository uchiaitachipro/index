"""
杂音检测 PyTorch CNN 独立训练脚本

特点：
1. 自动检测并使用最佳设备 (CUDA > MPS > CPU)
2. 支持进度条显示
3. 自动调整训练参数
4. Early Stopping 防止过拟合
5. 保存最佳模型
"""
import os
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['NUMBA_DISABLE_JIT'] = '0'

import numpy as np
import json
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
warnings.filterwarnings('ignore')

# 导入 PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("错误: 需要安装 PyTorch: pip install torch")
    exit(1)

# 导入 tqdm 进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("提示: 安装 tqdm 可显示进度条: pip install tqdm")


def get_torch_device():
    """
    获取最佳的 PyTorch 设备
    优先级: CUDA > MPS (Mac) > CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_device_name(device):
    """获取设备的友好名称"""
    if device.type == 'cuda':
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    elif device.type == 'mps':
        return "MPS (Apple Silicon GPU)"
    else:
        return "CPU"


class FocalLoss(nn.Module):
    """Focal Loss - 更好地处理样本不平衡问题"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # 聚焦参数，越大越关注难分类样本
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class ResidualBlock1D(nn.Module):
    """1D 残差块"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
        # 如果输入输出通道不同，需要1x1卷积调整
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + residual)
        return x


class AudioNoiseCNN(nn.Module):
    """改进版 1D CNN 模型 - 带残差连接和更深的网络"""
    
    def __init__(self, input_size):
        super().__init__()
        
        # 初始卷积
        self.conv_in = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn_in = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        
        # 残差块
        self.res1 = ResidualBlock1D(64, 64)
        self.res2 = ResidualBlock1D(64, 128)
        self.res3 = ResidualBlock1D(128, 256)
        self.res4 = ResidualBlock1D(256, 256)
        
        # 全局池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 全连接层 (使用avg和max池化的拼接)
        self.fc1 = nn.Linear(256 * 2, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        
        # Dropout
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.3)
        
    def forward(self, x):
        # x: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)
        
        # 初始卷积
        x = self.relu(self.bn_in(self.conv_in(x)))
        
        # 残差块
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        
        # 全局池化 (同时使用avg和max)
        avg_pool = self.global_avg_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        # 全连接
        x = self.dropout1(self.relu(self.bn_fc(self.fc1(x))))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x


class AudioNoiseCNNSimple(nn.Module):
    """简化版 CNN 模型（原版本，用于对比）"""
    
    def __init__(self, input_size):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class CNNModelWrapper:
    """CNN模型包装器，用于推理"""
    
    def __init__(self, model, scaler, device, input_size):
        self.model = model
        self.scaler = scaler
        self.device = device
        self.input_size = input_size
        self.classes_ = np.array([0, 1])
    
    def predict(self, X):
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()


def train_cnn(X_train, y_train, X_val=None, y_val=None, 
              epochs=100, lr=0.001, batch_size=None, 
              use_focal_loss=True, use_mixup=True, label_smoothing=0.1,
              model_type='advanced', verbose=True):
    """
    训练 PyTorch CNN 模型 (改进版)
    
    Args:
        X_train: 训练特征 (numpy array)
        y_train: 训练标签 (numpy array)
        X_val: 验证特征 (可选)
        y_val: 验证标签 (可选)
        epochs: 最大训练轮数
        lr: 初始学习率
        batch_size: 批次大小 (None则自动选择)
        use_focal_loss: 是否使用 Focal Loss (更好处理不平衡数据)
        use_mixup: 是否使用 Mixup 数据增强
        label_smoothing: 标签平滑系数 (0表示不使用)
        model_type: 模型类型 ('advanced' 或 'simple')
        verbose: 是否打印详细信息
    
    Returns:
        CNNModelWrapper: 训练好的模型包装器
    """
    device = get_torch_device()
    
    if verbose:
        print(f"\n{'='*60}")
        print("PyTorch CNN 训练 (改进版)")
        print(f"{'='*60}")
        print(f"  设备: {get_device_name(device)}")
        print(f"  模型类型: {model_type}")
        print(f"  训练样本数: {len(X_train)}")
        print(f"  特征数量: {X_train.shape[1]}")
        print(f"  正样本 (杂音): {np.sum(y_train)}")
        print(f"  负样本 (正常): {np.sum(1-y_train)}")
    
    # 根据设备类型自动调整参数
    if batch_size is None:
        batch_size = 64 if device.type in ('cuda', 'mps') else 32
    
    # 增加训练轮数以获得更好的收敛
    if device.type in ('cuda', 'mps'):
        lr = 0.002  # 稍微降低学习率
        epochs = min(epochs, 100)  # 增加到100轮
        early_stop_patience = 15  # 增加耐心
    else:
        early_stop_patience = 25
    
    if verbose:
        print(f"  批次大小: {batch_size}")
        print(f"  最大轮数: {epochs}")
        print(f"  初始学习率: {lr}")
        print(f"  Early Stop 耐心: {early_stop_patience}")
        print(f"  Focal Loss: {'✓' if use_focal_loss else '✗'}")
        print(f"  Mixup增强: {'✓' if use_mixup else '✗'}")
        print(f"  标签平滑: {label_smoothing}")
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 转换为张量
    X_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # 验证集
    if X_val is not None and y_val is not None:
        X_val_scaled = scaler.transform(X_val)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
        y_val_tensor = torch.LongTensor(y_val).to(device)
        has_validation = True
    else:
        has_validation = False
    
    # 创建模型
    if model_type == 'advanced':
        model = AudioNoiseCNN(X_train.shape[1]).to(device)
    else:
        model = AudioNoiseCNNSimple(X_train.shape[1]).to(device)
    
    # 计算类别权重（处理样本不平衡）- 增加杂音类的权重
    class_counts = np.bincount(y_train)
    # 使用更高的权重来减少漏检
    noise_weight = max(2.0, class_counts[0] / (class_counts[1] + 1) * 1.5)
    class_weights = torch.FloatTensor([1.0, noise_weight]).to(device)
    
    # 选择损失函数
    if use_focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        loss_name = "Focal Loss"
    elif label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        loss_name = f"CrossEntropy (平滑={label_smoothing})"
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        loss_name = "CrossEntropy"
    
    if verbose:
        print(f"  类别权重: [正常: 1.0, 杂音: {noise_weight:.2f}]")
        print(f"  损失函数: {loss_name}")
    
    # 使用 AdamW 优化器（更好的权重衰减）
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    
    # 使用余弦退火学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # 训练变量
    best_loss = float('inf')
    best_acc = 0.0
    patience_counter = 0
    best_model_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    
    if verbose:
        print(f"\n开始训练...")
    
    # 创建进度条
    if verbose and TQDM_AVAILABLE:
        epoch_pbar = tqdm(range(epochs), desc="  训练进度", unit="epoch",
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    else:
        epoch_pbar = range(epochs)
    
    for epoch in epoch_pbar:
        # 训练阶段
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            
            # Mixup 数据增强
            if use_mixup and np.random.random() > 0.5:
                lam = np.random.beta(0.4, 0.4)
                batch_size_curr = batch_x.size(0)
                index = torch.randperm(batch_size_curr).to(device)
                
                mixed_x = lam * batch_x + (1 - lam) * batch_x[index]
                outputs = model(mixed_x)
                
                # Mixup loss
                loss = lam * criterion(outputs, batch_y) + (1 - lam) * criterion(outputs, batch_y[index])
            else:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
            
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            # 计算准确率（使用原始数据，不是mixup后的）
            with torch.no_grad():
                orig_outputs = model(batch_x)
                _, predicted = torch.max(orig_outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        train_loss = total_loss / len(loader)
        train_acc = correct / total
        current_lr = optimizer.param_groups[0]['lr']
        
        # 更新学习率
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['lr'].append(current_lr)
        
        # 验证阶段
        if has_validation:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                _, val_predicted = torch.max(val_outputs, 1)
                val_acc = (val_predicted == y_val_tensor).sum().item() / len(y_val)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            monitor_loss = val_loss
        else:
            monitor_loss = train_loss
        
        # 更新进度条
        if verbose and TQDM_AVAILABLE:
            postfix = {
                'loss': f'{train_loss:.4f}',
                'acc': f'{train_acc:.2%}',
                'best': f'{best_loss:.4f}',
                'lr': f'{current_lr:.1e}'
            }
            if has_validation:
                postfix['val_acc'] = f'{val_acc:.2%}'
            epoch_pbar.set_postfix(postfix)
        
        # Early Stopping
        if monitor_loss < best_loss:
            best_loss = monitor_loss
            best_acc = train_acc
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                if verbose:
                    if TQDM_AVAILABLE:
                        epoch_pbar.close()
                    print(f"  Early stopping at epoch {epoch+1}")
                    print(f"  最佳 Loss: {best_loss:.4f}, 最佳 Acc: {best_acc:.2%}")
                break
        
        # 无 tqdm 时的简单进度显示
        if verbose and not TQDM_AVAILABLE and (epoch + 1) % 10 == 0:
            msg = f"  Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Acc: {train_acc:.2%}, LR: {current_lr:.1e}"
            if has_validation:
                msg += f", Val Acc: {val_acc:.2%}"
            print(msg)
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    
    if verbose:
        print(f"\n训练完成!")
        print(f"  最终 Loss: {best_loss:.4f}")
        print(f"  最终 Acc: {best_acc:.2%}")
    
    wrapper = CNNModelWrapper(model, scaler, device, X_train.shape[1])
    wrapper.history = history
    
    return wrapper


def save_model(wrapper, model_path='noise_detector_cnn.pth', info_path='noise_detector_cnn_info.json'):
    """
    保存 CNN 模型
    
    Args:
        wrapper: CNNModelWrapper 对象
        model_path: 模型保存路径
        info_path: 模型信息保存路径
    """
    # 保存 PyTorch 模型状态
    checkpoint = {
        'model_state_dict': wrapper.model.state_dict(),
        'scaler_mean': wrapper.scaler.mean_.tolist(),
        'scaler_scale': wrapper.scaler.scale_.tolist(),
        'input_size': wrapper.input_size,
    }
    torch.save(checkpoint, model_path)
    print(f"模型已保存到: {model_path}")
    
    # 保存模型信息
    info = {
        'model_type': 'AudioNoiseCNN',
        'input_size': wrapper.input_size,
        'device': str(wrapper.device),
        'history': wrapper.history if hasattr(wrapper, 'history') else {},
    }
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"模型信息已保存到: {info_path}")


def load_model(model_path='noise_detector_cnn.pth'):
    """
    加载 CNN 模型
    
    Args:
        model_path: 模型路径
    
    Returns:
        CNNModelWrapper: 加载的模型包装器
    """
    device = get_torch_device()
    checkpoint = torch.load(model_path, map_location=device)
    
    input_size = checkpoint['input_size']
    model = AudioNoiseCNN(input_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = StandardScaler()
    scaler.mean_ = np.array(checkpoint['scaler_mean'])
    scaler.scale_ = np.array(checkpoint['scaler_scale'])
    scaler.n_features_in_ = input_size
    
    wrapper = CNNModelWrapper(model, scaler, device, input_size)
    print(f"模型已从 {model_path} 加载，使用设备: {get_device_name(device)}")
    
    return wrapper


def evaluate_model(wrapper, X_test, y_test, file_names=None):
    """
    评估模型性能
    
    Args:
        wrapper: CNNModelWrapper 对象
        X_test: 测试特征
        y_test: 测试标签
        file_names: 文件名列表 (可选)
    """
    print(f"\n{'='*60}")
    print("模型评估")
    print(f"{'='*60}")
    
    y_pred = wrapper.predict(X_test)
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['无杂音', '有杂音']))
    
    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    
    # 错误分析
    if file_names is not None:
        errors = [(name, true, pred) for pred, true, name in zip(y_pred, y_test, file_names) if pred != true]
        if errors:
            print(f"\n错误分类的文件 ({len(errors)} 个):")
            false_negatives = [e for e in errors if e[1] == 1 and e[2] == 0]
            false_positives = [e for e in errors if e[1] == 0 and e[2] == 1]
            
            if false_negatives:
                print(f"\n  ❌ 杂音未识别 ({len(false_negatives)} 个):")
                for name, _, _ in false_negatives[:5]:
                    print(f"    {name}")
                if len(false_negatives) > 5:
                    print(f"    ... 还有 {len(false_negatives) - 5} 个")
            
            if false_positives:
                print(f"\n  ⚠️  误报 ({len(false_positives)} 个):")
                for name, _, _ in false_positives[:5]:
                    print(f"    {name}")
                if len(false_positives) > 5:
                    print(f"    ... 还有 {len(false_positives) - 5} 个")
        else:
            print("\n✓ 所有样本分类正确!")
    
    accuracy = np.mean(y_pred == y_test)
    return accuracy


# ============== 主程序 ==============
if __name__ == "__main__":
    print("="*60)
    print("杂音检测 PyTorch CNN 训练")
    print("="*60)
    
    # 检查环境
    print(f"\nPyTorch 版本: {torch.__version__}")
    device = get_torch_device()
    print(f"使用设备: {get_device_name(device)}")
    print(f"tqdm 进度条: {'✓' if TQDM_AVAILABLE else '✗'}")
    
    # 从 train_noise_detector_v5 导入数据准备函数
    try:
        from train_noise_detector_v5 import (
            prepare_training_data, 
            extract_features_v5,
            match_noise_file,
            FEATURE_NAMES_V5
        )
        print("✓ 成功导入特征提取模块")
    except ImportError as e:
        print(f"✗ 无法导入 train_noise_detector_v5: {e}")
        print("请确保 train_noise_detector_v5.py 在同一目录下")
        exit(1)
    
    # 读取杂音标签
    noise_txt = Path("./audio_noise_case/noise.txt")
    noise_files = set()
    if noise_txt.exists():
        with open(noise_txt, 'r') as f:
            noise_files = {line.strip() for line in f if line.strip()}
        print(f"\n已加载 {len(noise_files)} 个杂音文件标签")
    else:
        print(f"警告: 找不到 {noise_txt}")
    
    # 准备训练数据
    data_dir = Path("./audio_noise_case")
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        exit(1)
    
    print(f"\n正在提取特征...")
    X, y, file_names, matched_noise_files = prepare_training_data(
        data_dir, noise_files, use_augmentation=True, verbose=True
    )
    
    print(f"\n数据统计:")
    print(f"  总样本数: {len(X)} (含增强)")
    print(f"  特征数量: {X.shape[1]}")
    print(f"  有杂音样本: {np.sum(y)}")
    print(f"  无杂音样本: {np.sum(1-y)}")
    
    # 训练配置
    train_configs = [
        {
            'name': '改进版 (Focal Loss + Mixup)',
            'use_focal_loss': True,
            'use_mixup': True,
            'label_smoothing': 0.1,
            'model_type': 'advanced',
        },
        {
            'name': '改进版 (无Mixup)',
            'use_focal_loss': True,
            'use_mixup': False,
            'label_smoothing': 0.1,
            'model_type': 'advanced',
        },
    ]
    
    best_wrapper = None
    best_accuracy = 0
    best_errors = float('inf')
    best_config_name = ""
    
    for config in train_configs:
        print(f"\n{'#'*60}")
        print(f"# 配置: {config['name']}")
        print(f"{'#'*60}")
        
        # 训练模型
        wrapper = train_cnn(
            X, y, 
            epochs=100, 
            use_focal_loss=config['use_focal_loss'],
            use_mixup=config['use_mixup'],
            label_smoothing=config['label_smoothing'],
            model_type=config['model_type'],
            verbose=True
        )
        
        # 评估模型
        y_pred = wrapper.predict(X)
        accuracy = np.mean(y_pred == y)
        errors = np.sum(y_pred != y)
        
        print(f"\n  结果: 准确率={accuracy:.2%}, 错误数={errors}")
        
        # 选择最佳模型
        if errors < best_errors or (errors == best_errors and accuracy > best_accuracy):
            best_wrapper = wrapper
            best_accuracy = accuracy
            best_errors = errors
            best_config_name = config['name']
    
    print(f"\n{'='*60}")
    print(f"最佳配置: {best_config_name}")
    print(f"最佳准确率: {best_accuracy:.2%}")
    print(f"最少错误数: {best_errors}")
    print(f"{'='*60}")
    
    # 详细评估最佳模型
    evaluate_model(best_wrapper, X, y, file_names)
    
    # 保存最佳模型
    save_model(best_wrapper)
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)

