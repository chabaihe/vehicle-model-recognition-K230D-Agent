#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Stanford Cars 正式训练脚本：196 类细粒度车型分类

产出：
  1. 训练/验证曲线
  2. 测试集 Top-1 / Top-5 准确率
  3. 混淆矩阵 + **混淆结构分析**（哪些车型最容易互相认错）
  4. 每类准确率分布 + 最差类清单
  5. （可选）跨域测试：手机拍的真实车辆

用法：
    python train_cars.py --data data --epochs 15 --backbone resnet50
    python train_cars.py --data data --epochs 5  --backbone mobilenet_v3_large --quick
"""

from __future__ import annotations

import argparse
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
from torchvision import datasets, transforms, models
from torchvision.models import (resnet50, ResNet50_Weights, resnet18, ResNet18_Weights,
                               mobilenet_v3_large, MobileNet_V3_Large_Weights,
                               efficientnet_b0, EfficientNet_B0_Weights)

HERE = Path(__file__).resolve().parent
TOPIC = HERE.parent
RESULTS = TOPIC / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

N_CLASSES = 196
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


def find_splits(root: Path) -> tuple[Path, Path]:
    """定位 train / test 目录（兼容多种目录布局）。"""
    root = Path(root)
    for base in (root, *[p for p in root.iterdir() if p.is_dir()]):
        if not base.is_dir():
            continue
        train = next((d for d in base.iterdir() if d.is_dir() and d.name.lower() in
                      ("train", "training")), None)
        test = next((d for d in base.iterdir() if d.is_dir() and d.name.lower() in
                     ("test", "testing", "val", "valid")), None)
        if train and test:
            return train, test
    # 有些版本是 cars_train / cars_test 平铺
    for base in (root, *[p for p in root.iterdir() if p.is_dir()]):
        if not base.is_dir():
            continue
        subs = {d.name.lower(): d for d in base.iterdir() if d.is_dir()}
        if "cars_train" in subs and "cars_test" in subs:
            return subs["cars_train"], subs["cars_test"]
    raise SystemExit(f"[FAIL] 在 {root} 下找不到 train/test 目录")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(r"data"))
    ap.add_argument("--backbone", default="resnet50", choices=list(BACKBONES))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="只取部分数据快速验证")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 92)
    print(f"训练 | 骨干 {args.backbone} | {args.size}px | batch {args.batch} | {args.epochs} epoch")
    print(f"设备: {torch.cuda.get_device_name(0) if dev.type=='cuda' else 'CPU'}")
    print("=" * 92)

    train_dir, test_dir = find_splits(args.data)
    print(f"[数据] train={train_dir}\n       test ={test_dir}")

    # 按 ImageNet 统计量归一化（迁移学习标准做法）
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((args.size + 32, args.size + 32)),
        transforms.RandomCrop((args.size, args.size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])

    ds_tr = datasets.ImageFolder(str(train_dir), train_tf)
    ds_te = datasets.ImageFolder(str(test_dir), eval_tf)
    print(f"[数据] 训练 {len(ds_tr)} 张 / {len(ds_tr.classes)} 类 | 测试 {len(ds_te)} 张 / {len(ds_te.classes)} 类")

    # 类别分布（用于说明数据不均衡）
    counts = Counter(t for _, t in ds_tr.samples)
    n_per = np.array([counts.get(i, 0) for i in range(len(ds_tr.classes))])
    print(f"[数据] 每类样本数: 最少 {n_per.min()} / 最多 {n_per.max()} / 中位 {int(np.median(n_per))}")

    if args.quick:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(ds_tr), min(4000, len(ds_tr)), replace=False)
        ds_tr = Subset(ds_tr, idx)
        idx = rng.choice(len(ds_te), min(1000, len(ds_te)), replace=False)
        ds_te = Subset(ds_te, idx)
        print(f"[quick] 训练子集 {len(ds_tr)} / 测试子集 {len(ds_te)}")

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=True, persistent_workers=args.workers > 0)
    dl_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                       pin_memory=True)

    model = build(args.backbone).to(dev)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    hist = {"train_loss": [], "train_acc": [], "test_acc": [], "test_top5": []}
    best_acc = 0.0
    t_all = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        tl, correct, total = 0.0, 0, 0
        t0 = time.time()
        for x, y in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                out = model(x); loss = crit(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item(); total += y.size(0)
        sched.step()
        tr_loss, tr_acc = tl / total, correct / total

        # 评估
        model.eval()
        c1 = c5 = tot = 0
        with torch.no_grad():
            for x, y in dl_te:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                    out = model(x)
                c1 += (out.argmax(1) == y).sum().item()
                k = min(5, out.size(1))
                c5 += out.topk(k, dim=1).indices.eq(y.view(-1, 1)).any(1).sum().item()
                tot += y.size(0)
        acc1, acc5 = c1 / tot, c5 / tot
        hist["train_loss"].append(tr_loss); hist["train_acc"].append(tr_acc)
        hist["test_acc"].append(acc1); hist["test_top5"].append(acc5)
        peak = torch.cuda.max_memory_allocated(dev) / 1024**3 if dev.type == "cuda" else 0
        print(f"  epoch {ep:>3}/{args.epochs} | train_loss {tr_loss:.4f} | train_acc {tr_acc:.4f} "
              f"| test_top1 {acc1:.4f} | test_top5 {acc5:.4f} | {time.time()-t0:.0f}s | 峰值显存 {peak:.2f} GB",
              flush=True)
        if acc1 > best_acc:
            best_acc = acc1
            torch.save(model.state_dict(), RESULTS / f"best_{args.backbone}.pth")

    total_min = (time.time() - t_all) / 60
    print("=" * 92)
    print(f"【结果】最佳测试 Top-1 = {best_acc:.4f} | 最终 Top-5 = {hist['test_top5'][-1]:.4f}")
    print(f"        总训练耗时 {total_min:.1f} 分钟 | 峰值显存 "
          f"{torch.cuda.max_memory_allocated(dev)/1024**3:.2f} GB")
    # 基线对照：随机猜 / 最频类
    print(f"【对照】随机猜测 Top-1 = {1/N_CLASSES:.4f} | 最频类基线 = {n_per.max()/n_per.sum():.4f}")

    # ---------------- 混淆矩阵与混淆结构分析 ----------------
    print("\n[分析] 生成混淆矩阵与混淆结构...")
    model.load_state_dict(torch.load(RESULTS / f"best_{args.backbone}.pth", weights_only=True))
    model.eval()
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for x, y in dl_te:
            x = x.to(dev)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                p = model(x).argmax(1)
            for t, q in zip(y.numpy(), p.cpu().numpy()):
                cm[t, q] += 1

    classes = getattr(ds_te, "classes", None) or ds_te.dataset.classes
    per_class = cm.diagonal() / np.maximum(cm.sum(1), 1)
    order = np.argsort(per_class)
    print("\n  最差的 10 个类别（准确率最低）:")
    for i in order[:10]:
        print(f"    {per_class[i]:.3f}  {classes[i]}")
    print("\n  最好的 5 个类别:")
    for i in order[-5:][::-1]:
        print(f"    {per_class[i]:.3f}  {classes[i]}")

    # 混淆结构：找出最常被互相认错的类别对
    pairs = []
    for i in range(N_CLASSES):
        row = cm[i].copy(); row[i] = 0
        j = int(row.argmax())
        if row[j] > 0:
            pairs.append((int(row[j]), classes[i], classes[j], per_class[i]))
    pairs.sort(reverse=True)
    print("\n  最容易混淆的 10 对（真值 → 被认成）:")
    for cnt, a, b, acc in pairs[:10]:
        print(f"    {cnt:>3} 次  {a}  →  {b}   (该类准确率 {acc:.3f})")

    with open(RESULTS / f"test8_混淆对_{args.backbone}.csv", "w", encoding="utf-8-sig") as f:
        f.write("错误次数,真实类别,被误认成,该类准确率\n")
        for cnt, a, b, acc in pairs[:50]:
            f.write(f"{cnt},{a},{b},{acc:.4f}\n")

    # ---------------- 出图 ----------------
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(hist["train_loss"], label="train loss"); ax.set_xlabel("epoch"); ax.set_title("训练损失")
    ax.legend(); ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(hist["train_acc"], label="train top-1")
    ax.plot(hist["test_acc"], label="test top-1")
    ax.plot(hist["test_top5"], label="test top-5", ls="--")
    ax.axhline(1/N_CLASSES, color="gray", ls=":", label=f"随机 {1/N_CLASSES:.4f}")
    ax.set_xlabel("epoch"); ax.set_title("准确率曲线"); ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[0, 2])
    ax.bar(range(N_CLASSES), np.sort(per_class), color="#2980b9", width=1.0)
    ax.set_xlabel("类别（按准确率排序）"); ax.set_ylabel("准确率")
    ax.set_title(f"每类准确率分布（最差 {per_class.min():.2f} / 最佳 {per_class.max():.2f}）")
    ax.grid(alpha=.3)

    # 混淆矩阵（对数色标更易看清新颖结构）
    ax = fig.add_subplot(gs[1, :2])
    im = ax.imshow(np.log1p(cm), cmap="magma", aspect="auto")
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title("混淆矩阵（log(1+计数)）—— 明显亮块即易混淆类别群")
    fig.colorbar(im, ax=ax, fraction=0.025)

    ax = fig.add_subplot(gs[1, 2])
    top = pairs[:10][::-1]
    ax.barh([f"{a[:18]}\n→ {b[:18]}" for _, a, b, _ in top], [c for c, *_ in top], color="#c0392b")
    ax.set_xlabel("错误次数"); ax.set_title("最易混淆的 10 对车型"); ax.tick_params(labelsize=7)

    fig.suptitle(f"训练结果 · {args.backbone} · {args.size}px · {args.epochs} epoch · "
                 f"最佳Top-1={best_acc:.4f}", fontsize=14)
    fig.tight_layout()
    out = RESULTS / f"test8_训练结果_{args.backbone}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)

    with open(RESULTS / f"test8_训练摘要_{args.backbone}.json", "w", encoding="utf-8") as f:
        json.dump(dict(backbone=args.backbone, size=args.size, epochs=args.epochs,
                       batch=args.batch, best_top1=round(best_acc, 4),
                       final_top5=round(hist["test_top5"][-1], 4),
                       n_train=len(ds_tr), n_test=len(ds_te), n_classes=N_CLASSES,
                       minutes=round(total_min, 1),
                       peak_vram_gb=round(torch.cuda.max_memory_allocated(dev)/1024**3, 3)
                       if dev.type == "cuda" else 0,
                       random_baseline=round(1/N_CLASSES, 4),
                       worst_classes=[{"name": str(classes[i]), "acc": round(float(per_class[i]), 4)}
                                      for i in order[:10]],
                       top_confusions=[{"count": c, "true": a, "pred": b, "acc": round(acc, 4)}
                                       for c, a, b, acc in pairs[:10]]),
                      f, ensure_ascii=False, indent=2)
    print(f"\n[输出] {out}")
    print(f"[输出] {RESULTS / f'test8_训练摘要_{args.backbone}.json'}")
    print(f"[输出] {RESULTS / f'test8_混淆对_{args.backbone}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
