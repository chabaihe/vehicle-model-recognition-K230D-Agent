# K230D 部署代码

本目录包含**从 ONNX 到板端运行**的完整部署流程。

> ⭐⭐⭐⭐ 这是全项目**最难的部分**。建议先读完本 README 的「三个前置提醒」再动手。

---

## 三个前置提醒

| # | 提醒 | 后果 |
|:---:|---|---|
| **1** | **必须用独立的 Python 环境** | 训练环境（3.12/3.14）装不上 nncase，会一直报错 |
| **2** | **上板前先跑静态检查** | 否则要反复"改代码→拷卡→上板→发现报错" |
| **3** | **板卡必须按住 KEY2 启动** | 否则进图形桌面，IDE 连不上，串口无输出 |

---

## 脚本清单

### 电脑端（转换与验证）

| 脚本 | 运行环境 | 功能 |
|---|---|---|
| `convert_to_kmodel_car1000.py` | **独立 Python 3.10–3.12** | **核心**：ONNX → kmodel（量化编译） |
| `verify_kpu_plugin.py` | 同上 | 验证 nncase 的 k230 插件是否装好 |
| `verify_deploy.py` | 训练环境 | 上板前最后一道关卡：确认 ONNX 与 PyTorch 精度一致 |
| `verify_mean_scale_bug.py` | 训练环境 | 复现「归一化量纲」bug（本项目最隐蔽的问题） |
| `check_micropython_compat.py` | 任意 | **静态检查**：找出板端脚本里 MicroPython 不支持的写法 |
| `convert_to_kmodel.py` | 独立环境 | Stanford 版转换脚本（对照用） |

### 板端（CanMV IDE 里运行）

| 脚本 | 功能 |
|---|---|
| `k230_main.py` | **主程序**：摄像头采集 → 推理 → OSD 显示（28 FPS） |
| `k230_board_test.py` | 60 秒离线稳定性测试，报告写到 TF 卡 |
| `read_board_libs.py` | 读取板卡固件与库版本（排查环境问题） |

---

## 完整流程

### 第 1 步：创建独立的转换环境

**推荐：直接双击仓库根目录的 `install-k230.bat`**，它会自动建环境、装依赖并验证。

手动安装：

```bash
# 创建虚拟环境 —— 路径必须是纯英文！
python -m venv C:\vr_nncase_env
C:\vr_nncase_env\Scripts\activate

# 按清单安装（nncase 官方 wheel 只支持 cp39-cp313）
pip install -r requirements-k230.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 或用项目自带的脚本验证：
> `python check_env.py --k230`

**验证**：

```bash
python verify_kpu_plugin.py
```

应输出 k230 target 已注册。如果报 `KeyNotFoundException: 'k230' was not present`，
说明 **KPU 插件没装成功**。

> ⚠️ **三个常见问题**：
>
> **坑 1**：`nncase` 和 `nncase-kpu` 是**两个不同的包**，文件名只差一个下划线：
> - `nncase-2.11.0-cp312-cp312-win_amd64.whl`（18.75 MB）= **编译器本体**
> - `nncase_kpu-2.11.0-py2.py3-none-win_amd64.whl`（1.27 MB）= **KPU 插件** ← 容易漏
>
> **坑 2**：**PyPI 上没有 `nncase-kpu` 的 Windows wheel。**
> PyPI 只发布了 Linux 版（`manylinux2014_x86_64`），所以 Windows 上执行
> `pip install nncase-kpu==2.11.0` 会报
> `Could not find a version that satisfies the requirement ... (from versions: none)`。
> Windows 的 whl 只在 **GitHub Releases** 里。
> `requirements-k230.txt` 已用直链解决：
> ```
> nncase-kpu @ https://github.com/kendryte/nncase/releases/download/v2.11.0/nncase_kpu-2.11.0-py2.py3-none-win_amd64.whl ; sys_platform == "win32"
> ```
> 若直链也下不动，就手动到 GitHub Releases 页面下载上面两个 whl，再
> `pip install 路径\xxx.whl`。
>
> **坑 3**：**虚拟环境不能放在含中文的路径下。**
> nncase 是 .NET 库，`hostfxr` 加载器无法处理非 ASCII 路径，会直接报：
> ```
> RuntimeError: Failed to initialize hostfxr: 0x80008094.
> ```
> 所以 `install-k230.bat` 默认把环境建在 `%LOCALAPPDATA%\vr_nncase_env`。

### 第 2 步：转换模型

```bash
python convert_to_kmodel_car1000.py --onnx ..\05_模型文件\cars1000_resnet50_224.onnx
```

**转换做了什么**：

1. 图优化
2. **INT8 量化**（FP32 → INT8，体积压到 1/4）
3. 指定预处理参数

**产物**：`cars1000_resnet50_224.kmodel`（**25.87 MB**）

> 💡 本仓库 `05_模型文件/` 已包含这个文件，可直接使用。

**量化配置的关键点**：

| 参数 | 说明 |
|---|---|
| `input_type` | `uint8`（板端摄像头输出就是 uint8） |
| `input_range` | **必须与 mean/std 用同一数值尺度** |
| `mean` / `std` | ImageNet 标准值 `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]` |
| 校准集 | **必须是原始 uint8 数据**，且要覆盖足够多的类别 |

> ⚠️ **量纲配对是这个项目最隐蔽的坑**：
> `mean=[0.485,...]`（0–1 尺度）时，`input_range` 必须是 `[0,1]`，
> **不能写 `[0,255]`**。写错的后果是**编译不报错、运行不报错，但结果全错**。

### 第 3 步：静态检查板端代码

```bash
python check_micropython_compat.py k230_main.py
```

**为什么需要**：板端跑的是 **MicroPython**，它是 CPython 的精简子集，
很多常用功能不存在。首次上板时即遇到 `os.path.exists` 不存在的坑。

**已知的不兼容项**：

| CPython 写法 | MicroPython 替代 |
|---|---|
| `os.path.exists(p)` | `try: os.stat(p)` |
| `open(p, encoding="utf-8")` | `open(p, "r")` |
| `np.float32` | 用 `hasattr(np, "uint8")` 判断 |
| `np.argsort` 二维数组 | 用纯 Python `sorted()` |

### 第 4 步：拷贝到 TF 卡

把以下文件放到 **TF 卡的 `/sdcard/` 目录**（不是 BIN 分区）：

| 文件 | 说明 |
|---|---|
| `k230_main.py` → **重命名为 `main.py`** | 板卡上电自动运行 |
| `cars1000_resnet50_224.kmodel` | 模型（25.87 MB） |
| `class_names_cn.txt` | 1000 个类别的中文名称表 |

> ⚠️ TF 卡上的 `examples/`、`libs/`、`res/`、`micropython revision.txt`
> 是固件自带的，**不要动**。

> 💡 **建议买个读卡器**。用串口传 25.87 MB 的模型要一个多小时，用读卡器几秒钟。

### 第 5 步：上板运行

1. 插好 TF 卡
2. **按住开发板上的 KEY2 不放**
3. 按一下**复位**
4. 松开 KEY2
5. 打开 CanMV IDE，连接串口
6. 运行 `main.py`

> ⚠️ **第 2–4 步是关键**。K230D 默认从板载存储启动，
> 那套固件带图形桌面，不会把控制权交给 IDE，串口也没有输出。
>
> **替代方案**：用 K230BurningTool 把 `_tf.img` 烧到 EMMC，之后就不需要按键了。

**验证**：

板端 LCD 显示摄像头画面，车辆上有识别框：

| 框的颜色 | 含义 |
|---|---|
| 🟢 绿色 | 置信度 ≥ 50%，模型较确定 |
| 🟠 橙色 | 置信度 < 50%，模型不确定 |

期望帧率：**28 FPS**（显示分辨率 640×480）

---

## 结果全错、置信度 100%？看这里

**症状**：同一张图，电脑上识别正确，板子上完全错误，且**置信度恒定为 100%**。

**原因**：转换时归一化参数的**量纲配置错误**。

**为什么难排查**：编译不报错、运行不报错，只是结果不对。

**我们的四步定位法**（值得学习）：

| 步骤 | 做了什么 | 结论 |
|:---:|---|---|
| 1 | 电脑端用同一张图跑 PyTorch 与 ONNXRuntime 对拍 | 排除模型本身问题 |
| 2 | 在电脑上**人为构造各种预处理错误**，逐一模拟 | 发现只有"缺少归一化"会让置信度饱和到 100% |
| 3 | 检查板端源码的图像格式与通道顺序 | 确认无误 |
| 4 | 回到转换配置检查量纲 | **定位到问题** |

**核心结论**：

```python
# ❌ 错误：mean/std 是 0-1 尺度，input_range 却写了 0-255
input_range = [0, 255]
mean = [0.485, 0.456, 0.406]

# ✅ 正确：两者尺度一致
input_range = [0, 1]
mean = [0.485, 0.456, 0.406]
```

想复现这个 bug 看这里：`verify_mean_scale_bug.py`

---

## 板端程序说明

`k230_main.py` 使用正点原子 CanMV AI 开发框架的三个类：

| 类 | 职责 |
|---|---|
| `PipeLine` | 摄像头采集与显示通路 |
| `AIBase` | 模型加载与推理调度 |
| `Ai2d` | 图像预处理 |

**传感器通道配置**：

| 通道 | 格式 | 用途 |
|---|---|---|
| 通道 2 | `RGB_888_PLANAR` | 送入 AI 处理链路（注意 `swapRB=False`） |
| 通道 0 | YUV | 直接绑定显示层 |

**板端资源**：

| 资源 | 容量 |
|---|---|
| 运行内存 | 128 MB |
| 存储 | 2 GB |
| 模型占用 | 25.87 MB（20.2%） |

---

## 常见问题

### `KeyNotFoundException: 'k230' was not present in the dictionary`

KPU 插件没装。见第 1 步的「两个大坑」。

### `Dataset count 1 not equals to params 1 * samples 400`

校准集格式不对。必须用 `[(calib[i],) for i in range(len(calib))]` 这种带元组的形式。

### `need u8[1,3,224,224] but get f32`

校准数据必须是**原始 uint8**，不能是归一化后的浮点。

### `FormatException: 'np.float32(0.485)'`

numpy 2.x 的 `repr()` 问题。用 `[float(x) for x in MEAN]` 转成 Python float。

### IDE 一直"忙碌中"，没有 `>>>`

按 KEY2 + 复位，让板卡从 TF 卡启动。

### 板卡识别不出车

检查三点：
1. **视角** —— 训练图以正面和侧前方 45° 为主，正侧面识别率低
2. **车型是否在 1000 类中**
3. **归一化量纲是否正确**（置信度 100% 就是这个问题）

---

## 相关文档

- 上板完整实战记录：[`../08_技术文档/10_K230D上板实战指南.md`](../08_技术文档/10_K230D上板实战指南.md)
- 21 个踩坑汇总：[`../08_技术文档/01_问题与发现汇总.md`](../08_技术文档/01_问题与发现汇总.md)
- 置信度分析：[`../08_技术文档/12_置信度观察与结论.md`](../08_技术文档/12_置信度观察与结论.md)
