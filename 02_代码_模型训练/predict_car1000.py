#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Car-1000 车型识别推理工具 —— 喂一张图就出结果（中国车型，1000 类）

用法：
    # 单张图片
    python predict_car1000.py --img "D:\photos\我的车.jpg"

    # 整个文件夹（汇总成一张带预测结果的拼图）
    python predict_car1000.py --dir "D:\photos" --topk 5 --save

    # 指定权重
    python predict_car1000.py --img car.jpg --ckpt results/best_resnet50.pth

⚠️ 预处理必须与训练完全一致（Resize 224 → ToTensor → ImageNet 归一化）。
   本脚本已按训练配置实现，不要自行改动。

⚠️ 类别顺序以 **data/Car-1000/train 的目录顺序**为准，
   绝不能改用 JSON 顺序（主线 A 踩过"类别索引错位"的坑）。
"""

from __future__ import annotations

import argparse
import csv
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
from torchvision.models import (resnet50, resnet18, mobilenet_v3_large, efficientnet_b0)

HERE = Path(__file__).resolve().parent
TOPIC = HERE.parent
RESULTS = TOPIC / "results"
DATA = TOPIC / "data" / "Car-1000"

N_CLASSES = 1000
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


def load_labels() -> tuple[list[str], dict[str, str], dict[str, str]]:
    """
    返回 (类别ID顺序, 中文名映射, 大类映射)。
    顺序以 data/Car-1000/train 目录为准 —— 与 ImageFolder/模型输出索引一致。
    """
    order_file = RESULTS / "class_order.txt"
    tax = DATA / "class_taxonomy.csv"
    cn, big = {}, {}
    if tax.exists():
        with open(tax, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                cn[row["class_id"]] = row["name_cn"]
                big[row["class_id"]] = row["big_type"]

    if order_file.exists():
        ids = [l.strip() for l in order_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        d = DATA / "train"
        if not d.exists():
            raise SystemExit(f"[FAIL] 找不到类别目录: {d}")
        ids = sorted(x.name for x in d.iterdir() if x.is_dir())
        order_file.write_text("\n".join(ids), encoding="utf-8")
        print(f"[类别] 已固化类别顺序到 {order_file}（{len(ids)} 类）")
    return ids, cn, big


@torch.no_grad()
def predict(model, img_path: Path, size: int, topk: int, dev):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    im = Image.open(img_path).convert("RGB")
    x = tf(im).unsqueeze(0).to(dev)
    prob = torch.softmax(model(x), dim=1)[0]
    p, idx = prob.topk(topk)
    return [(int(i), float(v)) for v, i in zip(p.cpu(), idx.cpu())], im


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=Path, help="单张图片路径")
    ap.add_argument("--dir", type=Path, help="图片文件夹")
    ap.add_argument("--backbone", default="resnet50", choices=list(BUILDERS))
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--size", type=int, default=224, help="必须与训练一致")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--save", action="store_true", help="保存可视化结果")
    args = ap.parse_args()

    if not args.img and not args.dir:
        raise SystemExit("请指定 --img 或 --dir（-h 查看帮助）")

    ckpt = args.ckpt or (RESULTS / f"best_{args.backbone}.pth")
    if not ckpt.exists():
        raise SystemExit(f"[FAIL] 权重不存在: {ckpt}\n       可用: "
                         f"{[p.name for p in RESULTS.glob('best_*.pth')]}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids, cn, big = load_labels()
    if len(ids) != N_CLASSES:
        print(f"[警告] 类别数 {len(ids)} ≠ {N_CLASSES}，请确认权重与类别表匹配")

    model = build_model(args.backbone).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    model.eval()

    print("=" * 88)
    print(f"Car-1000 车型识别 · {args.backbone} · {ckpt.name} · {args.size}px · 设备 {dev}")
    print(f"类别数 {len(ids)} | 预处理 ImageNet mean/std（与训练一致）")
    print("=" * 88)

    if args.img:
        files = [args.img]
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = sorted(p for p in Path(args.dir).rglob("*") if p.suffix.lower() in exts)
        if not files:
            raise SystemExit(f"[FAIL] {args.dir} 下没有图片")
        print(f"文件夹模式：共 {len(files)} 张图\n")

    panels = []
    for f in files:
        if not f.exists():
            print(f"[跳过] 不存在: {f}")
            continue
        preds, im = predict(model, f, args.size, args.topk, dev)
        print(f"\n【{f.name}】")
        for rank, (ci, prob) in enumerate(preds, 1):
            cid = ids[ci] if ci < len(ids) else "?"
            name = cn.get(cid, cid)
            bt = big.get(cid, "")
            bar = "█" * max(1, int(prob * 40))
            star = "  ← 最可能" if rank == 1 else ""
            print(f"  Top{rank}  {prob*100:6.2f}%  {name:<34}[{bt}]{bar}{star}")
        if args.save:
            panels.append((f, im, [(cn.get(ids[c], ids[c]), p) for c, p in preds]))

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
        fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.2 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax, (f, im, preds) in zip(axes, panels[:n]):
            ax.imshow(im); ax.axis("off")
            txt = "\n".join(f"{i+1}. {c[:30]}  {p*100:.1f}%" for i, (c, p) in enumerate(preds))
            ax.set_title(f"{f.name}\n{txt}", fontsize=9)
        for ax in axes[len(panels[:n]):]:
            ax.axis("off")
        fig.suptitle(f"Car-1000 车型识别结果 · {args.backbone}", fontsize=13)
        fig.tight_layout()
        out = RESULTS / "predict_result.png"
        fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"\n[输出] {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
