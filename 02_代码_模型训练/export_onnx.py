#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
K230 部署脚本（阶段 1）：导出 ResNet-50 的 ONNX

在**当前训练环境（Python 3.14 + torch）**里运行，产出 cars196.onnx。
之后再到独立的 Python 3.12 环境里用 nncase 转成 .kmodel（见 convert_to_kmodel.py）。

用法：
    python export_onnx.py
    python export_onnx.py --backbone resnet50 --size 224
    python export_onnx.py --dynamic-batch      # 一般不需要；板端是单张推理

要点（都是踩过的坑）：
  1. **静态 shape 更好**：nncase 对固定输入 shape 的优化更充分，板端也是单张推理；
  2. **opset 用 17**：不要用 18+，nncase 未必跟进；
  3. **导出前 eval()**：否则 BatchNorm 用训练态统计量，结果会错；
  4. 导出后建议用 onnx-simplifier 过一遍（可选，能去掉残留节点）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
import torch.nn as nn
from torchvision.models import (resnet50, resnet18, mobilenet_v3_large,
                                efficientnet_b0)

HERE = Path(__file__).resolve().parent
TOPIC = HERE.parent
RESULTS = TOPIC / "results"
DEPLOY = TOPIC / "deploy"          # 部署产物集中放这里
DEPLOY.mkdir(parents=True, exist_ok=True)

N_CLASSES = 196
BUILDERS = {
    "resnet50": resnet50,
    "resnet18": resnet18,
    "mobilenet_v3_large": mobilenet_v3_large,
    "efficientnet_b0": efficientnet_b0,
}


def build_model(backbone: str) -> nn.Module:
    m = BUILDERS[backbone](weights=None)
    if hasattr(m, "fc"):
        m.fc = nn.Linear(m.fc.in_features, N_CLASSES)
    elif hasattr(m, "classifier"):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, N_CLASSES)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50", choices=list(BUILDERS))
    ap.add_argument("--size", type=int, default=224, help="必须与训练分辨率一致")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dynamic-batch", action="store_true",
                    help="导出动态 batch（板端单张推理时不需要）")
    ap.add_argument("--simplify", action="store_true", help="用 onnx-simplifier 优化")
    args = ap.parse_args()

    ckpt = RESULTS / f"best_{args.backbone}.pth"
    if not ckpt.exists():
        raise SystemExit(f"[FAIL] 权重不存在: {ckpt}\n       可用: "
                         f"{[p.name for p in RESULTS.glob('best_*.pth')]}")

    print("=" * 84)
    print(f"导出 ONNX · 骨干 {args.backbone} · 输入 {args.size}px · opset {args.opset}")
    print("=" * 84)

    model = build_model(args.backbone)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()                                   # ← 必须，否则 BN 统计量错误
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[模型] 参数 {n_param/1e6:.2f} M · 权重 {ckpt.name}")

    out = DEPLOY / f"cars196_{args.backbone}_{args.size}.onnx"
    dummy = torch.randn(1, 3, args.size, args.size)
    dyn = {"input": {0: "batch"}, "logits": {0: "batch"}} if args.dynamic_batch else None

    with torch.no_grad():
        torch.onnx.export(
            model, dummy, str(out),
            input_names=["input"], output_names=["logits"],
            opset_version=args.opset, dynamo=False,
            dynamic_axes=dyn,
        )

    # 校验 ONNX 可加载且输出维度正确
    try:
        import onnx
        m = onnx.load(str(out))
        onnx.checker.check_model(m)
        print(f"[校验] ONNX checker 通过")
    except ImportError:
        print("[校验] 未安装 onnx，跳过检查（pip install onnx）")
    except Exception as e:
        print(f"[校验] ⚠️ ONNX 检查失败: {str(e)[:120]}")

    if args.simplify:
        try:
            import onnxsim
            from onnxsim import simplify
            import onnx
            simplified, ok = simplify(onnx.load(str(out)))
            if ok:
                out_s = out.with_name(out.stem + "_sim.onnx")
                onnx.save(simplified, str(out_s))
                print(f"[简化] 已输出 {out_s.name}（{out_s.stat().st_size/1024**2:.2f} MB）")
                out = out_s
            else:
                print("[简化] onnx-simplifier 未能简化，保留原文件")
        except ImportError:
            print("[简化] 未安装 onnxsim（pip install onnxsim），跳过")

    mb = out.stat().st_size / 1024**2
    print(f"\n[输出] {out}")
    print(f"       体积 {mb:.2f} MB · 输入 {[1,3,args.size,args.size]} · 输出 [1,{N_CLASSES}]")

    # 同时导出类别表，供板端使用
    cls_txt = DEPLOY / "class_names.txt"
    d = Path(r"data/prepared/test")
    if d.exists():
        names = sorted(x.name for x in d.iterdir() if x.is_dir())
        cls_txt.write_text("\n".join(names), encoding="utf-8")
        print(f"[输出] {cls_txt}  （{len(names)} 类，顺序与模型输出索引严格对应）")

    print("\n[下一步] 到独立 Python 3.12 环境运行:")
    print(f"        python convert_to_kmodel.py --onnx \"{out}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
