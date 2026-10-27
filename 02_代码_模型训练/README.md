# 模型训练代码

本目录包含**从数据准备到模型导出**的完整训练流程。

---

## 运行环境

本目录所有脚本共用**主环境**，依赖清单在仓库根目录的 `requirements.txt`。

```bash
# 一键安装（推荐）：双击仓库根目录的 install.bat

# 手动安装：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 自检：
python check_env.py
```

| 依赖 | 用途 |
|---|---|
| `torch` + `torchvision` | 训练与预训练 ResNet |
| `numpy` | 数值计算 |
| `pillow` | 读写图片 |
| `matplotlib` | 训练曲线、预测结果图 |
| `scipy` | `prepare_cars.py` 用 |
| `onnx` / `onnxruntime` | 导出 ONNX 并对拍验证 |
| `onnxsim`（可选） | `export_onnx.py --simplify` 时用 |

> 💡 **PyTorch 必须装 GPU 版**。直接 `pip install torch` 在 Windows 上会得到
> CPU 版，训练慢几十倍。用 `install.bat` 会自动判断。

---

## 脚本清单

### Car-1000 主线（推荐）

按编号顺序执行：

| 顺序 | 脚本 | 功能 | 主要参数 |
|:---:|---|---|---|
| 1 | `prepare_car1000.py` | 数据准备：整理为 ImageFolder 结构 | `--root` 原始目录 `--out` 输出目录 |
| 2 | `bench_car1000.py` | 训练前实测：每轮耗时与显存峰值 | `--data` 数据目录 |
| 3 | `train_car1000.py` | 正式训练（1000 类） | `--data` `--epochs` `--batch-size` |
| 4 | `evaluate_car1000.py` | 分层评估与混淆结构分析 | `--ckpt` 权重 `--data` 数据目录 |
| 5 | `predict_car1000.py` | 推理：单张图片或整个文件夹 | `--image` 或 `--dir` |
| 6 | `export_onnx_car1000.py` | 导出 ONNX + 与 PyTorch 对拍 | `--ckpt` 权重 |

### Stanford Cars 基线（第一阶段对照）

| 脚本 | 功能 |
|---|---|
| `prepare_cars.py` | 数据准备（Stanford Cars 版） |
| `train_cars.py` | 训练 196 类基线模型 |
| `analyze_confusion.py` | 混淆结构分析：把「哪些车认错」提升为「为什么认错」 |
| `export_onnx.py` | 导出 Stanford 版 ONNX |

> 💡 Stanford 版是**数据集选择论证的证据**。没有它就无法说明「为什么必须换数据集」。

---

## 完整用法

### 第 1 步：准备数据

```bash
python prepare_car1000.py --root "D:\data\Car-1000-dataset" --out "D:\data\Car1000_split"
```

**做什么**：把按编号存放的原始数据整理成 PyTorch 需要的 ImageFolder 结构。
整理后应得到 1000 个以车型命名的文件夹。

**注意**：脚本使用硬链接（同一分区时），不会额外占用磁盘空间。

**期望输出**：

```
训练集 83,756 张（59.7%）
验证集 27,658 张（19.7%）
测试集 28,853 张（20.6%）
类别数 1000
```

### 第 2 步：实测训练成本

```bash
python bench_car1000.py --data "D:\data\Car1000_split"
```

**为什么先做这一步**：8.3 万张训练图，如果不先测，可能跑到一半发现要 20 小时。

**实测结果**（RTX 4060 Laptop 8 GB）：

| 指标 | 数值 |
|---|---:|
| 每轮耗时 | 约 6.5 分钟 |
| 峰值显存 | 3.05 GB |
| 建议总轮数 | 15 轮（约 97 分钟） |

### 第 3 步：训练

```bash
python train_car1000.py --data "D:\data\Car1000_split" --epochs 15 --batch 64
```

**关键配置**（脚本内已设好）：

| 配置 | 取值 | 为什么 |
|---|---|---|
| 骨干网络 | ResNet-50 | 量化友好（见下方说明） |
| 输入尺寸 | 224×224 | 标准尺寸 |
| 优化器 | AdamW，lr 3e-4 | 迁移学习常用 |
| 学习率调度 | 余弦退火 | 后期自动降 lr，收敛更稳 |
| 损失函数 | 交叉熵 + 类别加权 + 标签平滑 0.1 | 应对类别不均衡（31:1） |
| 数据增强 | 随机裁剪、水平翻转、颜色抖动 | 抑制过拟合 |

**为什么用 ResNet-50 而不是更小的 MobileNetV3？**

两者的量化损失实测对比如下：

| 骨干网络 | 浮点精度 | INT8 量化后 | 损失 |
|---|---:|---:|---:|
| MobileNetV3-Large | 83.52% | 66.19% | **−17.33 pp** ❌ |
| ResNet-50 | 91.33% | 90.46% | **−0.87 pp** ✅ |

端侧部署必须量化，所以**更大的 ResNet-50 反而更划算**。

**产物**：`results/best_resnet50.pth`（约 97 MB）

### 第 4 步：评估

```bash
python evaluate_car1000.py --ckpt results/best_resnet50.pth --split test
```

**输出三层指标**：

| 指标 | 我们的结果 |
|---|---:|
| Top-1（具体车型） | **86.77%** |
| Top-5 | 98.16% |
| **车辆大类** | **98.24%** |

**还会做混淆结构分析**，核心结论：

> 最严重的 100 组误认中，**100% 发生在同品牌内部，跨品牌误认为 0 次**。

主要失败模式是**同款车不同动力形式**（燃油 / 插混 / 纯电），
这些车外观差异在 224 分辨率下不足十个像素。

### 第 5 步：推理

```bash
# 单张图片
python predict_car1000.py --img "D:\photos\my_car.jpg"

# 整个文件夹
python predict_car1000.py --dir "D:\photos" --save
```

### 第 6 步：导出 ONNX

```bash
python export_onnx_car1000.py --ckpt results/best_resnet50.pth
```

**这一步会做对拍验证**：用同一批图片分别跑 PyTorch 与 ONNXRuntime，逐张比对。

**必须通过的标准**：`一致率: 100.00%`

> ⚠️ 如果不是 100%，**先别往下做部署**。否则上板后发现结果不对，
> 你会以为是 nncase 的问题，白白排查很久。

**产物**：`cars1000_resnet50_224.onnx`（97.41 MB）

---

## 数据流

```
原始数据集
    │  prepare_car1000.py
    ▼
ImageFolder 结构（train / val / test）
    │  train_car1000.py
    ▼
best_resnet50.pth  ──evaluate_car1000.py──▶  评估结果
    │
    │  export_onnx_car1000.py
    ▼
cars1000_resnet50_224.onnx  ──▶  （交给 03_代码_K230部署）
```

---

## 常见问题

### `CUDA out of memory`

把 `--batch-size` 从 64 改成 32 或 16。还不行就把输入尺寸降到 192。

### 准确率明显低于 86.77%

按可能性排序检查：

1. **没加载 ImageNet 预训练权重**（影响最大）
2. **数据划分不是官方的**（自己随机切分会破坏可比性）
3. **训练轮数不够**
4. **类别映射顺序错了** —— 这个是隐蔽的坑，会导致推理正常但标签整体错位

### 训练太慢

确认 PyTorch 是 CUDA 版（`torch.cuda.is_available()` 为 `True`）。
CPU 训练会慢几十倍。

---

## 相关文档

- 训练与评估细节：[`../08_技术文档/07_Car-1000训练与评估结果.md`](../08_技术文档/07_Car-1000训练与评估结果.md)
- 两个数据集的对比：[`../08_技术文档/08_两数据集模型结果汇总.md`](../08_技术文档/08_两数据集模型结果汇总.md)
- 为什么换数据集：[`../08_技术文档/13_技术路线思考.md`](../08_技术文档/13_技术路线思考.md)
