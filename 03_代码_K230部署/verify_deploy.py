#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
端到端验证：部署用的 ONNX 模型精度是否与 PyTorch 一致

用途：上板前最后一道关卡。若这里对不上，说明导出或预处理有问题，
      绝不要急着做 kmodel —— 否则会以为是 nncase 的问题，白排查很久。

做法：用同一批测试图，分别跑
      ① PyTorch 权重
      ② 部署目录里的 cars196_*.onnx（onnxruntime，CPU）
    比较两者的 Top-1 是否一致。

用法：
    python verify_deploy.py --onnx ..\deploy\cars196_resnet50_224.onnx --n 800
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, resnet18, mobilenet_v3_large, efficientnet_b0

HERE = Path(__file__).resolve().parent
TOPIC = HERE.parent
N_CLASSES = 196
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
BUILDERS = {"resnet50": resnet50, "resnet18": resnet18,
            "mobilenet_v3_large": mobilenet_v3_large,
            "efficientnet_b0": efficientnet_b0}


def build_model(backbone: str) -> nn.Module:
    m = BUILDERS[backbone](weights=None)
    if hasattr(m, "fc"):
        m.fc = nn.Linear(m.fc.in_features, N_CLASSES)
    elif hasattr(m, "classifier"):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, N_CLASSES)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--backbone", default="resnet50", choices=list(BUILDERS))
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--data", type=Path, default=Path(r"data/prepared/test"))
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--n", type=int, default=800)
    args = ap.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("需要 onnxruntime: python -m pip install onnxruntime")

    ckpt = args.ckpt or (HERE.parent / "results" / f"best_{args.backbone}.pth")
    if not ckpt.exists():
        raise SystemExit(f"[FAIL] 权重不存在: {ckpt}")

    # 收集测试样本
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in args.data.rglob("*") if p.suffix.lower() in exts]
    classes = sorted(x.name for x in args.data.iterdir() if x.is_dir())
    c2i = {c: i for i, c in enumerate(classes)}
    rng = np.random.default_rng(0)
    sel = rng.choice(len(files), min(args.n, len(files)), replace=False)

    tf = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    print("=" * 84)
    print(f"端到端验证 · {args.onnx.name}  vs  {ckpt.name}")
    print(f"测试样本 {len(sel)} 张 · 覆盖类别 {len({files[i].parent.name for i in sel})}")
    print("=" * 84)

    # PyTorch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.backbone).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    model.eval()

    # ONNX（需拷贝到 ASCII 路径，onnxruntime 不认中文路径）
    import shutil, tempfile
    wd = Path(tempfile.gettempdir()) / "verify_deploy"
    wd.mkdir(parents=True, exist_ok=True)
    onnx_ascii = wd / args.onnx.name
    shutil.copy2(args.onnx, onnx_ascii)
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(str(onnx_ascii), so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    ys = np.array([c2i[files[i].parent.name] for i in sel])
    xs = np.stack([tf(Image.open(files[i]).convert("RGB")).numpy() for i in sel])

    # PyTorch 预测
    with torch.no_grad():
        pt_out = []
        for k in range(0, len(xs), 32):
            xb = torch.from_numpy(xs[k:k + 32]).to(dev)
            pt_out.append(model(xb).cpu().numpy())
        pt_out = np.concatenate(pt_out)
    pt_top1 = (pt_out.argmax(1) == ys).mean()

    # ONNX 预测（导出的模型是静态 batch=1，与 nncase 最匹配，因此逐张推理）
    ort_out = np.zeros_like(pt_out)
    for k in range(len(xs)):
        ort_out[k] = sess.run(None, {inp: xs[k:k + 1]})[0][0]
    ort_top1 = (ort_out.argmax(1) == ys).mean()

    agree = (pt_out.argmax(1) == ort_out.argmax(1)).mean()
    max_diff = float(np.abs(pt_out - ort_out).max())

    print(f"  PyTorch  Top-1 : {pt_top1:.4f}")
    print(f"  ONNX     Top-1 : {ort_top1:.4f}")
    print(f"  预测一致率     : {agree*100:.2f}%")
    print(f"  logits 最大差异: {max_diff:.6f}")
    print("=" * 84)

    if agree > 0.995 and abs(pt_top1 - ort_top1) < 0.01:
        print("[✅ 通过] ONNX 与 PyTorch 一致 —— 可以放心进入 nncase 转 kmodel 环节。")
        return 0
    print("[❌ 未通过] 两者不一致！先查这里，不要去动 nncase：")
    print("   1) 导出时是否调用了 model.eval()？（BatchNorm 统计量）")
    print("   2) 归一化 mean/std 是否与训练一致？")
    print("   3) 是否误加了预处理（onnx 里不该再有归一化）？")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
