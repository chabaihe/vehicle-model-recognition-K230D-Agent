#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
K230 部署脚本（阶段 2）：ONNX → .kmodel（nncase）

⚠️ 本脚本必须在**独立的 Python 3.10/3.12 环境**里运行！
   nncase 官方 wheel 只覆盖 cp39–cp313，且对 numpy/onnx 版本敏感。
   不要用训练用的 Python 3.14 环境。

环境准备（一次性）：
    # 1) 装一个 Python 3.12（本机当前是 3.14，nncase 不支持）
    #    https://www.python.org/downloads/release/python-3129/
    py -3.12 -m venv C:\vr_nncase_env
    C:\vr_nncase_env\Scripts\activate

    # 2) 装 nncase（注意 nncase 与 nncase-kpu 是两个包，都要装）
    pip install nncase==2.11.0 nncase-kpu==2.11.0

    # 3) 装转换依赖（按官方文档版本，可省数小时排错）
    pip install "numpy==1.19.5" "onnx==1.9.0" "onnxruntime==1.8.0"
    pip install onnx-simplifier==0.3.6 onnxoptimizer==0.2.6

    # 4) 还需要 .NET 7 运行时（nncase 的 IR 依赖）：
    #    https://dotnet.microsoft.com/download/dotnet/7.0

用法：
    python convert_to_kmodel.py --onnx ..\deploy\cars196_resnet50_224.onnx
    python convert_to_kmodel.py --onnx ... --samples 300 --calib-dir data/prepared/train

关键参数说明：
    --samples   校准样本数，建议 200–300，且**必须覆盖足够多的类别**（见避坑说明）
    --method    量化校准方法，v2.11 默认 Kld（KL 散度），比 NoClip 更能保持分类精度
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

# 与训练完全一致
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_calibration(calib_dir: Path, size: int, n: int, seed: int = 0) -> np.ndarray:
    """
    生成校准数据集。**必须覆盖尽量多的类别**——只取同一类会让量化尺度严重失真。
    返回形状 (n, 1, 3, size, size) 的 float32 数组（nncase 要求 5 维：NCHW + batch 占位）。
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in calib_dir.rglob("*") if p.suffix.lower() in exts]
    if not files:
        raise SystemExit(f"[FAIL] {calib_dir} 下没有图片")

    rng = random.Random(seed)
    # 按类别分层抽样：先按父目录分组，再从每组取，保证覆盖度
    by_cls: dict[str, list[Path]] = {}
    for p in files:
        by_cls.setdefault(p.parent.name, []).append(p)
    cls_list = sorted(by_cls)
    rng.shuffle(cls_list)

    picked: list[Path] = []
    i = 0
    while len(picked) < n and cls_list:
        c = cls_list[i % len(cls_list)]
        pool = by_cls[c]
        picked.append(rng.choice(pool))
        i += 1

    print(f"[校准] 从 {len(by_cls)} 个类别中分层抽取 {len(picked)} 张（覆盖 "
          f"{len({p.parent.name for p in picked})} 类）")

    arr = np.zeros((len(picked), 1, 3, size, size), dtype=np.float32)
    for k, p in enumerate(picked):
        im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
        a = (a - MEAN) / STD
        arr[k, 0] = a.transpose(2, 0, 1)
    return arr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--method", default="Kld", choices=["Kld", "NoClip", "MinMax"])
    ap.add_argument("--calib-dir", type=Path, default=Path(r"data/prepared/train"))
    ap.add_argument("--input-type", default="uint8", choices=["uint8", "float32"],
                    help="uint8 与摄像头采集一致，省一次板端转换")
    ap.add_argument("--dry-run", action="store_true", help="只做检查，不真正编译")
    args = ap.parse_args()

    if not args.onnx.exists():
        raise SystemExit(f"[FAIL] ONNX 不存在: {args.onnx}")

    print("=" * 88)
    print("K230 模型转换 · nncase → .kmodel")
    print("=" * 88)

    # ---- 环境自检（这是最容易翻车的一步）----
    print("[环境自检]")
    print(f"  Python {sys.version.split()[0]}")
    try:
        import nncase
        print(f"  nncase {getattr(nncase, '__version__', '未知')}")
    except ImportError:
        print("  ❌ nncase 未安装！")
        print("     → 必须在独立 Python 3.10/3.12 环境里运行本脚本：")
        print("       py -3.12 -m venv C:\\vr_nncase_env")
        print("       C:\\vr_nncase_env\\Scripts\\activate")
        print("       pip install nncase==2.11.0 nncase-kpu==2.11.0")
        return 2
    import onnx
    print(f"  onnx {onnx.__version__}")
    print(f"  numpy {np.__version__}")

    # ---- 检查 ONNX 输入输出 ----
    m = onnx.load(str(args.onnx))
    onnx.checker.check_model(m)
    inp = m.graph.input[0]
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"[ONNX] 输入 {inp.name} shape={dims} · 体积 {args.onnx.stat().st_size/1024**2:.2f} MB")
    if dims[0] not in (1, 0):
        print(f"  ⚠️ batch 维不是 1（{dims[0]}），nncase 编译可能失败，建议重新导出为静态 batch=1")

    if args.dry_run:
        calib = build_calibration(args.calib_dir, args.size, min(args.samples, 20))
        print(f"[dry-run] 校准集形状 {calib.shape}，未执行编译")
        return 0

    calib = build_calibration(args.calib_dir, args.size, args.samples)

    out_path = args.out or (DEPLOY / (args.onnx.stem + ".kmodel"))

    compile_options = nncase.CompileOptions()
    compile_options.target = "k230"                  # K230D 同属 k230 目标
    compile_options.input_type = args.input_type
    compile_options.input_layout = "NCHW"
    compile_options.output_layout = "NCHW"
    compile_options.input_shape = [1, 3, args.size, args.size]
    if args.input_type == "uint8":
        # 预处理写进图里：板端只喂原始像素，省一次转换、也避免漏掉
        compile_options.preprocess = True
        compile_options.input_range = [0, 255]
        compile_options.mean = list(MEAN)
        compile_options.std = list(STD)
        compile_options.swapRB = False               # 板端按 RGB 喂；若用 BGR 则改 True

    ptq = nncase.PTQTensorOptions()
    ptq.samples_count = len(calib)
    ptq.calibrate_method = args.method
    ptq.set_tensor_data([(calib,)])

    print(f"[编译] target=k230 · input={args.input_type} · shape={compile_options.input_shape}")
    print(f"[量化] 方法={args.method} · 校准样本={len(calib)}")

    compiler = nncase.Compiler(compile_options)
    compiler.import_onnx(args.onnx.read_bytes(), nncase.ImportOptions())
    compiler.use_ptq(ptq)
    compiler.compile()

    kmodel = compiler.gencode_tobytes()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(kmodel)

    mb = len(kmodel) / 1024**2
    print("\n" + "=" * 88)
    print(f"[完成] {out_path}")
    print(f"       .kmodel 体积 {mb:.2f} MB  （占 K230D 128MB DDR 的 {mb/128*100:.1f}%）")

    # 同时拷类别表（板端要按索引读类别名）
    cls_src = DEPLOY / "class_names.txt"
    if cls_src.exists():
        (DEPLOY / "class_names_for_sd.txt").write_text(
            cls_src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[完成] {DEPLOY / 'class_names_for_sd.txt'}  ← 拷到 TF 卡")

    print("\n[下一步] 把这两个文件拷到 TF 卡根目录：")
    print(f"       {out_path.name}         （模型）")
    print(f"       class_names_for_sd.txt  （类别表）")
    print("       然后用 k230_main.py 作为 main.py 上板运行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
