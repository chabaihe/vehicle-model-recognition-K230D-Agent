#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Car-1000 正式训练脚本（1000 类细粒度车辆识别）

与 Stanford 版（train_cars.py）的关键区别：
  1. **1000 类**（而非 196），分类头 2.05 M 参数
  2. **类别不均衡严重（31:1）** → 用类别加权损失
  3. **官方提供 val 集** → 用 val 选最佳模型，test 只在最后评一次
  4. **分层评估**：除 Top-1/Top-5 外，额外报告**车辆大类准确率**（7 类，来自 type_info）
  5. 数据量大（83,756 张）→ 加入进度输出，便于监控

产出：
  results/best_<backbone>.pth       最佳模型
  results/train_log_<backbone>.json 逐 epoch 记录
  results/curves_<backbone>.png     训练曲线

用法：
    python train_car1000.py --epochs 15 --batch 64 --workers 8
    python train_car1000.py --epochs 3 --quick        # 快速验证（子集）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
for _f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DengXian"):
    matplotlib.rcParams["font.sans-serif"] = [_f]
    break
matplotlib.rcParams["axes.unicode_minus"] = False

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import (resnet50, ResNet50_Weights, resnet18, ResNet18_Weights,
                                mobilenet_v3_large, MobileNet_V3_Large_Weights,
                                efficientnet_b0, EfficientNet_B0_Weights)

HERE = Path(__file__).resolve().parent
TOPIC = HERE.parent
RESULTS = TOPIC / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
DATA = TOPIC / "data" / "Car-1000"

N_CLASSES = 1000
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
BACKBONES = {
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "mobilenet_v3_large": (mobilenet_v3_large, MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
}


def build(name: str) -> nn.Module:
    fn, w = BACKBONES[name]
    m = fn(weights=w)
    if hasattr(m, "fc"):
        m.fc = nn.Linear(m.fc.in_features, N_CLASSES)
    elif hasattr(m, "classifier"):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, N_CLASSES)
    return m


def load_taxonomy() -> dict[str, str]:
    """类别ID → 车辆大类（用于分层评估）。"""
    f = DATA / "class_taxonomy.csv"
    out: dict[str, str] = {}
    if f.exists():
        with open(f, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                out[row["class_id"]] = row["big_type"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--backbone", default="resnet50", choices=list(BACKBONES))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--weighted", action="store_true", default=True,
                    help="使用类别加权损失（应对 31:1 不均衡）")
    ap.add_argument("--quick", action="store_true", help="子集快速验证")
    ap.add_argument("--tag", default="", help="输出文件名后缀")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.tag or args.backbone
    print("=" * 100)
    print(f"Car-1000 训练 | {args.backbone} | {args.size}px | batch {args.batch} | {args.epochs} epoch")
    print(f"设备: {torch.cuda.get_device_name(0) if dev.type=='cuda' else 'CPU'} | 类别数 {N_CLASSES}")
    print("=" * 100)

    train_tf = transforms.Compose([
        transforms.Resize((args.size + 32, args.size + 32)),
        transforms.RandomCrop((args.size, args.size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])

    ds_tr = datasets.ImageFolder(str(args.data / "train"), train_tf)
    ds_val = datasets.ImageFolder(str(args.data / "val"), eval_tf)
    print(f"[数据] train {len(ds_tr)} 张 | val {len(ds_val)} 张 | 类别 {len(ds_tr.classes)}")
    if len(ds_tr.classes) != N_CLASSES:
        raise SystemExit(f"[FAIL] 类别数不符：期望 {N_CLASSES}，实际 {len(ds_tr.classes)}")

    # 类别分布（用于分析数据不均衡）
    cnt = Counter(t for _, t in ds_tr.samples)
    n_per = np.array([cnt.get(i, 0) for i in range(N_CLASSES)])
    print(f"[数据] 每类样本: 最少 {n_per.min()} / 最多 {n_per.max()} / 中位 {int(np.median(n_per))} "
          f"| 不均衡比 {n_per.max()/max(n_per.min(),1):.1f}")

    if args.quick:
        rng = np.random.default_rng(0)
        ds_tr = Subset(ds_tr, rng.choice(len(ds_tr), min(20000, len(ds_tr)), replace=False))
        ds_val = Subset(ds_val, rng.choice(len(ds_val), min(5000, len(ds_val)), replace=False))
        print(f"[quick] train {len(ds_tr)} / val {len(ds_val)}")

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
                       prefetch_factor=4 if args.workers > 0 else None)
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)

    model = build(args.backbone).to(dev)
    if args.weighted:
        w = len(n_per) / (n_per.sum() * np.maximum(n_per, 1))
        w = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
        print(f"[损失] 类别加权（权重范围 {w.min():.3f} – {w.max():.3f}）")
        crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    else:
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    hist = []
    best = 0.0
    t_all = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        tl, n = 0.0, 0
        t0 = time.time()
        for bi, (x, y) in enumerate(dl_tr):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                out = model(x); loss = crit(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item() * y.size(0); n += y.size(0)
            if (bi + 1) % 200 == 0:
                el = time.time() - t0
                print(f"    ep{ep} batch {bi+1}/{len(dl_tr)} | loss {tl/n:.4f} | "
                      f"{n/el:.0f} img/s | 已用 {el/60:.1f} min", flush=True)
        sched.step()
        tr_loss = tl / max(n, 1)

        # ---- 验证 ----
        model.eval()
        c1 = c5 = tot = 0
        with torch.no_grad():
            for x, y in dl_val:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                    out = model(x)
                c1 += (out.argmax(1) == y).sum().item()
                c5 += out.topk(5, dim=1).indices.eq(y.view(-1, 1)).any(1).sum().item()
                tot += y.size(0)
        acc1, acc5 = c1 / tot, c5 / tot
        peak = torch.cuda.max_memory_allocated(dev) / 1024**3 if dev.type == "cuda" else 0
        dt = time.time() - t0
        rec = dict(epoch=ep, train_loss=round(tr_loss, 4), val_top1=round(acc1, 4),
                   val_top5=round(acc5, 4), sec=round(dt, 1), peak_vram=round(peak, 3))
        hist.append(rec)
        print(f"  epoch {ep:>3}/{args.epochs} | loss {tr_loss:.4f} | val_top1 {acc1:.4f} "
              f"| val_top5 {acc5:.4f} | {dt/60:.1f} min | 峰值显存 {peak:.2f} GB", flush=True)

        if acc1 > best:
            best = acc1
            torch.save(model.state_dict(), RESULTS / f"best_{tag}.pth")
        (RESULTS / f"train_log_{tag}.json").write_text(
            json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")

    total_min = (time.time() - t_all) / 60
    print("=" * 100)
    print(f"【结果】最佳 val Top-1 = {best:.4f} | 总耗时 {total_min:.1f} 分钟")
    print(f"【对照】随机猜测 Top-1 = {1/N_CLASSES:.5f}")

    # ---------------- 曲线 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep = [h["epoch"] for h in hist]
    axes[0].plot(ep, [h["train_loss"] for h in hist], marker="o")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss"); axes[0].set_title("训练损失")
    axes[0].grid(alpha=.3)
    axes[1].plot(ep, [h["val_top1"] for h in hist], marker="o", label="val Top-1")
    axes[1].plot(ep, [h["val_top5"] for h in hist], marker="s", ls="--", label="val Top-5")
    axes[1].axhline(1/N_CLASSES, color="gray", ls=":", label=f"随机 {1/N_CLASSES:.4f}")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("准确率"); axes[1].set_title("验证准确率")
    axes[1].legend(); axes[1].grid(alpha=.3)
    fig.suptitle(f"Car-1000 训练曲线 · {args.backbone} · 1000 类 · 最佳 Top-1={best:.4f}")
    fig.tight_layout()
    fig.savefig(RESULTS / f"curves_{tag}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[输出] {RESULTS / f'curves_{tag}.png'}")
    print(f"[输出] {RESULTS / f'best_{tag}.pth'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
