#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Car-1000 部署脚本（阶段 2）：ONNX → .kmodel（nncase）

⚠️ 本脚本必须在**独立的 Python 3.10/3.12 环境**里运行！
   nncase 官方 wheel 只覆盖 cp39–cp313，且对 numpy/onnx 版本敏感。
   **不要用训练用的 Python 3.14 环境。**

环境准备（一次性）：
    py -3.12 -m venv C:\vr_nncase_env
    C:\vr_nncase_env\Scripts\activate
    pip install nncase==2.11.0 nncase-kpu==2.11.0
    pip install "numpy==1.19.5" "onnx==1.9.0" "onnxruntime==1.8.0"
    pip install onnx-simplifier==0.3.6 onnxoptimizer==0.2.6
    # 另需 .NET 7 运行时：https://dotnet.microsoft.com/download/dotnet/7.0

用法：
    python convert_to_kmodel_car1000.py --onnx ..\deploy\cars1000_resnet50_224.onnx
    python convert_to_kmodel_car1000.py --onnx ... --samples 400 --dry-run
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
TOPIC = HERE.parent
DEPLOY = TOPIC / "deploy"
DATA = TOPIC / "data" / "Car-1000"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_calibration(calib_dir: Path, size: int, n: int, seed: int = 0) -> np.ndarray:
    """
    **分层抽样**生成校准集：必须覆盖尽量多的类别。
    只用少数类别的图会让量化尺度严重偏向那些类，其他类精度会塌。
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in calib_dir.rglob("*") if p.suffix.lower() in exts]
    if not files:
        raise SystemExit(f"[FAIL] {calib_dir} 下没有图片")

    rng = random.Random(seed)
    by_cls: dict[str, list[Path]] = {}
    for p in files:
        by_cls.setdefault(p.parent.name, []).append(p)
    cls_list = sorted(by_cls)
    rng.shuffle(cls_list)

    picked: list[Path] = []
    i = 0
    while len(picked) < n and cls_list:
        c = cls_list[i % len(cls_list)]
        picked.append(rng.choice(by_cls[c]))
        i += 1

    covered = len({p.parent.name for p in picked})
    print(f"[校准] 从 {len(by_cls)} 类中分层抽取 {len(picked)} 张（覆盖 {covered} 类）")

    # ⚠️ 关键：因为下面设了 co.preprocess = True，
    # nncase 会在编译时用 co.mean / co.std 自己做归一化，
    # 所以校准数据必须是**原始 uint8 像素**，不能提前归一化。
    # 若这里传归一化后的 float32，会报：
    #     Feed Value Is Invalid, need u8[1,3,224,224] but get f32[1,3,224,224]!
    arr = np.zeros((len(picked), 1, 3, size, size), dtype=np.uint8)
    for k, p in enumerate(picked):
        im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        arr[k, 0] = np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)
    return arr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--method", default="Kld", choices=["Kld", "NoClip", "MinMax"])
    ap.add_argument("--calib-dir", type=Path, default=DATA / "train")
    ap.add_argument("--input-type", default="uint8", choices=["uint8", "float32"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.onnx.exists():
        raise SystemExit(f"[FAIL] ONNX 不存在: {args.onnx}")

    print("=" * 92)
    print("Car-1000 模型转换 · nncase → .kmodel")
    print("=" * 92)
    print("[环境自检]")
    print(f"  Python {sys.version.split()[0]}")
    try:
        import nncase
        print(f"  nncase {getattr(nncase, '__version__', '未知')}")
    except ImportError:
        print("  ❌ nncase 未安装！必须用独立 Python 3.10/3.12 环境：")
        print("     py -3.12 -m venv C:\\vr_nncase_env")
        print("     C:\\vr_nncase_env\\Scripts\\activate")
        print("     pip install nncase==2.11.0 nncase-kpu==2.11.0")
        return 2
    import onnx
    print(f"  onnx {onnx.__version__} | numpy {np.__version__}")

    m = onnx.load(str(args.onnx))
    onnx.checker.check_model(m)
    inp = m.graph.input[0]
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"[ONNX] 输入 {inp.name} shape={dims} | {args.onnx.stat().st_size/1024**2:.2f} MB")
    if dims[0] not in (1, 0):
        print(f"  ⚠️ batch 维不是 1（{dims[0]}），建议重新导出为静态 batch=1")

    if args.dry_run:
        c = build_calibration(args.calib_dir, args.size, min(args.samples, 20))
        print(f"[dry-run] 校准集形状 {c.shape}，未编译")
        return 0

    calib = build_calibration(args.calib_dir, args.size, args.samples)
    out_path = args.out or (DEPLOY / (args.onnx.stem + ".kmodel"))

    co = nncase.CompileOptions()
    co.target = "k230"
    co.input_type = args.input_type
    co.input_layout = "NCHW"
    co.output_layout = "NCHW"
    co.input_shape = [1, 3, args.size, args.size]
    if args.input_type == "uint8":
        co.preprocess = True
        # ⚠️⚠️ 关键修正（v2）—— mean/std 必须与 input_range 同量纲！
        #
        # 官方文档：input_range 决定「反量化」后 float 的取值范围：
        #     input_range=[0,255] → 反量化只做类型转换，数值仍是 0~255
        #     input_range=[0,1]   → 反量化直接把定点数映射到 0~1
        # 而随后的 Normalization 用的是 co.mean / co.std，**必须与之一致**。
        #
        # v1 的错误：input_range=[0,255] 却给了 0~1 量纲的 ImageNet 系数
        #     → 减 0.485 约等于「不减」，除 0.229 约等于「放大 4.4 倍」
        #     → 输入变成 0~1118 → logits 爆炸 → softmax 饱和 100%
        #     → 板端表现为「置信度 100% 但类别完全错误」
        #
        # v2 修正：input_range=[0,1]，mean/std 保持 ImageNet 的 0~1 系数，
        #          使 kmodel 内部预处理与训练/PC 端严格一致。
        co.input_range = [0, 1]
        co.mean = [float(x) for x in MEAN]      # 0.485 / 0.456 / 0.406
        co.std = [float(x) for x in STD]        # 0.229 / 0.224 / 0.225
        co.swapRB = False          # 通道2 是 RGB888_PLANAR；若用 BGR 则改 True

    ptq = nncase.PTQTensorOptions()
    ptq.samples_count = len(calib)
    ptq.calibrate_method = args.method
    # ⚠️ 校准数据格式很重要！
    # calib 形状为 (N, 1, 3, H, W) —— 已经带了 dataset 维度。
    # nncase 期望 [dataset][每输入一张图]，所以必须解包成 N 个单元素元组：
    #     [(sample0,), (sample1,), ...]
    # 若写成 [(calib,)] 或 [calib]，会报
    #     "Dataset count 1 not equals to params count 1 * samples count N"
    ptq.set_tensor_data([(calib[i],) for i in range(len(calib))])

    print(f"[编译] target=k230 | input={args.input_type} | shape={co.input_shape}")
    print(f"[量化] 方法={args.method} | 校准样本={len(calib)}")

    compiler = nncase.Compiler(co)
    compiler.import_onnx(args.onnx.read_bytes(), nncase.ImportOptions())
    compiler.use_ptq(ptq)
    compiler.compile()

    kmodel = compiler.gencode_tobytes()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(kmodel)
    mb = len(kmodel) / 1024**2

    print("\n" + "=" * 92)
    print(f"[完成] {out_path}")
    print(f"       .kmodel 体积 {mb:.2f} MB  （占 K230D 128MB DDR 的 {mb/128*100:.1f}%）")
    print(f"\n[下一步] 把以下文件拷到 TF 卡根目录：")
    print(f"       {out_path.name}")
    print(f"       class_names_cn.txt  （或 class_names_en.txt，板端无中文字体时用）")
    print(f"       k230_main.py  →  重命名为 main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
