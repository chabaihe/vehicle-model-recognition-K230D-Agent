#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
车型识别推理工具 —— 直接用你训练好的 ResNet-50 模型预测任意一张图片

用法：
    # 单张图片
    python predict_cars.py --img "D:\photos\car.jpg"

    # 整个文件夹（会汇总成一张带预测结果的拼图）
    python predict_cars.py --dir "D:\photos"

    # 显示 Top-5 候选（默认 Top-3）
    python predict_cars.py --img car.jpg --topk 5

    # 用 MobileNetV3 或指定权重
    python predict_cars.py --img car.jpg --backbone mobilenet_v3_large

    # 保存可视化结果
    python predict_cars.py --img car.jpg --save

⚠️ 重要：预处理必须与训练完全一致（Resize 到 224 → ToTensor → ImageNet 归一化），
   否则精度会大幅下降。本脚本已按训练配置实现，不要自行改动。
"""

from __future__ import annotations

import argparse
import json
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
from torchvision.models import (resnet50, resnet18, mobilenet_v3_large,
                                efficientnet_b0)

HERE = Path(__file__).resolve().parent
TOPIC = HERE.parent
RESULTS = TOPIC / "results"

N_CLASSES = 196
# 与 train_cars.py 完全一致的归一化参数
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

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


def load_classes() -> list[str]:
    """
    返回**与模型输出索引严格对应**的类别名列表。

    ⚠️ 关键坑：ImageFolder 是按**目录名字母序**给类别编号的，而
       Stanford Cars 原始 class_names.txt 是**官方标注顺序**，两者不同！
       （例如 'Audi S4 Sedan 2012' 在原始表里是第 22 个，字母序里是第 17 个）
       因此必须用 ImageFolder 的实际顺序，否则预测结果会整体错位。
    """
    d = Path(r"data/prepared/test")
    if d.exists():
        dirs = sorted(x.name for x in d.iterdir() if x.is_dir())
        if dirs:
            return dirs
    d2 = Path(r"data/prepared/train")
    if d2.exists():
        dirs = sorted(x.name for x in d2.iterdir() if x.is_dir())
        if dirs:
            return dirs
    raise SystemExit("[FAIL] 找不到类别目录，请先运行 prepare_cars.py")


def load_display_names() -> dict[str, str]:
    """目录名 → 原始类名 的映射（仅用于显示，不参与索引）。"""
    f = Path(r"data/prepared/class_names.txt")
    m: dict[str, str] = {}
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                m[parts[2]] = parts[1]
    return m


@torch.no_grad()
def predict(model: nn.Module, img_path: Path, size: int, topk: int, dev):
    tf = transforms.Compose([
        transforms.Resize((size, size)),          # 与训练评估一致
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    im = Image.open(img_path).convert("RGB")
    x = tf(im).unsqueeze(0).to(dev)
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0]
    p, idx = prob.topk(topk)
    return [(int(i), float(v)) for v, i in zip(p.cpu(), idx.cpu())], im


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=Path, help="单张图片路径")
    ap.add_argument("--dir", type=Path, help="图片文件夹")
    ap.add_argument("--backbone", default="resnet50", choices=list(BUILDERS))
    ap.add_argument("--ckpt", type=Path, default=None, help="自定义权重路径")
    ap.add_argument("--size", type=int, default=224, help="必须与训练一致")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--save", action="store_true", help="保存可视化结果到 results/")
    args = ap.parse_args()

    if not args.img and not args.dir:
        raise SystemExit("请指定 --img 或 --dir（-h 查看帮助）")

    ckpt = args.ckpt or (RESULTS / f"best_{args.backbone}.pth")
    if not ckpt.exists():
        raise SystemExit(f"[FAIL] 权重不存在: {ckpt}\n       可用的: "
                         f"{[p.name for p in RESULTS.glob('best_*.pth')]}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes()
    model = build_model(args.backbone).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    model.eval()

    print("=" * 78)
    print(f"车型识别 · 骨干 {args.backbone} · 权重 {ckpt.name} · 输入 {args.size}px · 设备 {dev}")
    print(f"类别数 {len(classes)} | 预处理 ImageNet mean/std（与训练一致）")
    print("=" * 78)

    if args.img:
        files = [args.img]
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = sorted(p for p in Path(args.dir).rglob("*") if p.suffix.lower() in exts)
        if not files:
            raise SystemExit(f"[FAIL] {args.dir} 下没有图片")
        print(f"文件夹模式：共 {len(files)} 张图\n")

    panels = []
    for i, f in enumerate(files):
        if not f.exists():
            print(f"[跳过] 不存在: {f}")
            continue
        preds, im = predict(model, f, args.size, args.topk, dev)
        print(f"\n【{f.name}】")
        for rank, (ci, prob) in enumerate(preds, 1):
            bar = "█" * max(1, int(prob * 40))
            star = "  ← 最可能" if rank == 1 else ""
            print(f"  Top{rank}  {prob*100:6.2f}%  {classes[ci]:<42}{bar}{star}")
        if args.save:
            panels.append((f, im, [(classes[c], p) for c, p in preds]))

    if args.save and panels:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for _f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DengXian"):
            matplotlib.rcParams["font.sans-serif"] = [_f]
            break
        matplotlib.rcParams["axes.unicode_minus"] = False

        n = min(len(panels), 8)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax, (f, im, preds) in zip(axes, panels[:n]):
            ax.imshow(im); ax.axis("off")
            txt = "\n".join(f"{i+1}. {c[:28]}  {p*100:.1f}%" for i, (c, p) in enumerate(preds))
            ax.set_title(f"{f.name}\n{txt}", fontsize=9)
        for ax in axes[len(panels[:n]):]:
            ax.axis("off")
        fig.suptitle(f"车型识别结果 · {args.backbone}", fontsize=13)
        fig.tight_layout()
        out = RESULTS / "test8_推理结果.png"
        fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"\n[输出] {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
