# 杂音检测模型训练脚本依赖说明

## 第三方依赖库列表

训练脚本 `train_noise_detector.py` 需要以下第三方 Python 库：

### 必需依赖 (Required)

1. **numpy** (>=1.26.2)
   - 用途：数值计算、数组操作
   - 安装：`pip install numpy` 或 `pip install numpy>=1.26.2`

2. **librosa** (>=0.10.2)
   - 用途：音频处理、特征提取（STFT、MFCC、频谱特征等）
   - 安装：`pip install librosa` 或 `pip install librosa>=0.10.2`
   - 注意：librosa 可能还需要 `soundfile` 作为音频文件读取的后端

3. **scikit-learn** (>=1.3.0)
   - 用途：机器学习模型（RandomForestClassifier）、数据划分、评估指标
   - 安装：`pip install scikit-learn` 或 `pip install scikit-learn>=1.3.0`
   - 包含模块：
     - `sklearn.ensemble.RandomForestClassifier`
     - `sklearn.model_selection.train_test_split`
     - `sklearn.metrics.classification_report`, `confusion_matrix`

4. **joblib** (>=1.3.0)
   - 用途：模型序列化，保存和加载训练好的模型
   - 安装：`pip install joblib` 或 `pip install joblib>=1.3.0`
   - 注意：scikit-learn 通常会包含 joblib，但建议显式安装

### 标准库 (Built-in, 无需安装)

以下库是 Python 标准库，无需额外安装：
- `pathlib` - 文件路径操作
- `json` - JSON 数据处理

## 安装方法

### 方法 1: 使用 requirements 文件

```bash
pip install -r requirements_noise_detector.txt
```

### 方法 2: 单独安装

```bash
pip install numpy>=1.26.2 librosa>=0.10.2 scikit-learn>=1.3.0 joblib>=1.3.0
```

### 方法 3: 使用项目已有的依赖

如果项目已经安装了 `numpy` 和 `librosa`（通常在项目主依赖中），只需要额外安装：

```bash
pip install scikit-learn joblib
```

## 检查安装

运行以下命令检查所有依赖是否已安装：

```python
python -c "import numpy; import librosa; import sklearn; import joblib; print('所有依赖已安装！')"
```

## 依赖关系说明

- **librosa** 依赖 **soundfile** 或 **audioread** 来读取音频文件
- **scikit-learn** 依赖 **numpy** 和 **scipy**
- **joblib** 通常随 **scikit-learn** 一起安装

## 版本兼容性

- Python 版本：>= 3.10（根据项目要求）
- 所有库都支持 Python 3.10+

## 故障排除

如果遇到导入错误：

1. **librosa 导入错误**：
   ```bash
   pip install soundfile  # 音频文件读取后端
   ```

2. **scikit-learn 导入错误**：
   ```bash
   pip install scikit-learn scipy  # scipy 是 scikit-learn 的依赖
   ```

3. **joblib 导入错误**：
   ```bash
   pip install joblib
   ```

