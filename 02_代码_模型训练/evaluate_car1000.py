#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Car-1000 分层评估与混淆结构分析

**这是主线 B 的核心分析**，利用 Car-1000 的**三级层次标签**，
做比 Stanford 版更进一步的评估：

  ① 最终测试集评估（Top-1 / Top-5）—— 只在最后评一次，不参与调参
  ② **分层评估**：7 类车辆大类的准确率（车型大类远比具体车型容易）
  ③ **层次混淆矩阵**：把 1000×1000 混淆矩阵聚合到 7×7 大类
  ④ 每类准确率分布 + 最差/最好类别
  ⑤ **混淆结构分析**：同品牌内部 vs 跨品牌
  ⑥ 少样本类别表现（对应 31:1 不均衡）

产出（results/）：
  eval_car1000.json              全部指标
  eval_per_class.csv             每类准确率
  eval_confusion_pairs.csv       最易混淆的类别对
  eval_hierarchy.png             分层评估与混淆矩阵图

用法：
    python evaluate_car1000.py --ckpt results/best_resnet50.pth
    python evaluate_car1000.py --split val          # 用 val 快速看
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (resnet50, resnet18, mobilenet_v3_large, efficientnet_b0)

HERE = Path(__file__).resolve().parent
TOPIC = HERE.parent
RESULTS = TOPIC / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
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


def load_taxonomy() -> tuple[list[str], dict[str, str], dict[str, str]]:
    """返回 (类ID顺序, 类ID→中文名, 类ID→大类)。顺序以 train 目录为准。"""
    d = DATA / "train"
    ids = sorted(x.name for x in d.iterdir() if x.is_dir())
    cn, big = {}, {}
    f = DATA / "class_taxonomy.csv"
    if f.exists():
        with open(f, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                cn[row["class_id"]] = row["name_cn"]
                big[row["class_id"]] = row["big_type"]
    return ids, cn, big


def load_counts() -> dict[str, int]:
    out = {}
    f = DATA / "class_taxonomy.csv"
    if f.exists():
        with open(f, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                out[row["class_id"]] = int(row["n_train"] or 0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=RESULTS / "best_resnet50.pth")
    ap.add_argument("--backbone", default="resnet50", choices=list(BUILDERS))
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print(f"Car-1000 分层评估 | {args.backbone} | 权重 {args.ckpt.name} | 划分 {args.split}")
    print(f"设备: {torch.cuda.get_device_name(0) if dev.type=='cuda' else 'CPU'}")
    print("=" * 100)

    if not args.ckpt.exists():
        raise SystemExit(f"[FAIL] 权重不存在: {args.ckpt}")

    ids, cn, big = load_taxonomy()
    counts = load_counts()
    n_cls = len(ids)
    print(f"[类别] {n_cls} 个（顺序以 train 目录为准，与模型输出索引对应）")

    tf = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    ds = datasets.ImageFolder(str(DATA / args.split), tf)
    if len(ds.classes) != n_cls:
        print(f"[警告] 目录类别数 {len(ds.classes)} ≠ 期望 {n_cls}")
    if ds.classes != ids:
        print(f"[警告] ImageFolder 类别顺序与 train 目录不一致！"
              f"\n        前 3: {ds.classes[:3]} vs {ids[:3]}")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                    pin_memory=True, persistent_workers=args.workers > 0)
    print(f"[数据] {args.split}: {len(ds)} 张")

    model = build_model(args.backbone).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=True))
    model.eval()

    # ---- 推理：收集 all logits（1000 类）----
    all_logits, all_y = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(dl):
            x = x.to(dev, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
                out = model(x)
            all_logits.append(out.float().cpu())
            all_y.append(y)
            if (i + 1) % 100 == 0:
                print(f"  推理 {i+1}/{len(dl)} batch", flush=True)

    logits = torch.cat(all_logits)
    y = torch.cat(all_y).numpy()
    prob = torch.softmax(logits, dim=1)
    top1 = logits.argmax(1).numpy()
    top5 = logits.topk(5, dim=1).indices.numpy()

    acc1 = float((top1 == y).mean())
    acc5 = float((top5 == y[:, None]).any(1).mean())
    print(f"\n【① 总体】{args.split} Top-1 = {acc1:.4f} | Top-5 = {acc5:.4f} "
          f"| 随机基线 {1/n_cls:.5f}")

    # ---- ② 分层评估（7 类大类）----
    big_of = np.array([big.get(ids[i], "未知") for i in range(n_cls)])
    y_big, p_big = big_of[y], big_of[top1]
    print(f"\n【② 分层评估】按车辆大类")
    print(f"  {'大类':<10}{'样本数':>8}{'类别数':>8}{'Top-1':>10}")
    rows_big = []
    for b in sorted(set(big_of.tolist())):
        m = y_big == b
        if m.sum() == 0:
            continue
        a = float((p_big[m] == y_big[m]).mean())
        n_cls_b = int((big_of == b).sum())
        rows_big.append((b, int(m.sum()), n_cls_b, a))
        print(f"  {b:<10}{m.sum():>8}{n_cls_b:>8}{a:>10.4f}")
    # 大类整体准确率（把预测映射到大类后比较）
    big_acc = float((p_big == y_big).mean())
    print(f"  {'全部':<10}{len(y):>8}{n_cls:>8}{big_acc:>10.4f}   ← 大类层面的准确率")

    # ---- ③ 层次混淆矩阵（7×7）----
    labels_big = sorted(set(big_of.tolist()))
    idx = {b: i for i, b in enumerate(labels_big)}
    cm_big = np.zeros((len(labels_big), len(labels_big)), dtype=np.int64)
    for t, p in zip(y_big, p_big):
        cm_big[idx[t], idx[p]] += 1
    cm_big_pct = cm_big / np.maximum(cm_big.sum(1, keepdims=True), 1) * 100

    # ---- ④ 每类准确率 ----
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, p in zip(y, top1):
        cm[t, p] += 1
    per_class = cm.diagonal() / np.maximum(cm.sum(1), 1)
    order = np.argsort(per_class)
    print(f"\n【④ 每类准确率】最低 {per_class.min():.3f} / 最高 {per_class.max():.3f} "
          f"/ 中位 {np.median(per_class):.3f}")
    print(f"  最差的 8 个类别:")
    for i in order[:8]:
        print(f"    {per_class[i]:.3f}  [{ids[i]}] {cn.get(ids[i],'?')}  (train {counts.get(ids[i],0)} 张)")
    print(f"  最好的 5 个类别:")
    for i in order[-5:][::-1]:
        print(f"    {per_class[i]:.3f}  [{ids[i]}] {cn.get(ids[i],'?')}")

    # ---- ⑤ 混淆结构分析 ----
    pairs = []
    for i in range(n_cls):
        row = cm[i].copy(); row[i] = 0
        j = int(row.argmax())
        if row[j] > 0:
            pairs.append((int(row[j]), ids[i], ids[j], cn.get(ids[i], ""), cn.get(ids[j], "")))
    pairs.sort(reverse=True)
    print(f"\n【⑤ 最易混淆的 12 对】")
    for c, ai, bi, an, bn in pairs[:12]:
        print(f"  {c:>3} 次  {an}  →  {bn}")

    # 同品牌 vs 跨品牌
    same_brand = diff_brand = 0
    for c, ai, bi, an, bn in pairs[:100]:
        b1 = an.split("_")[0] if "_" in an else an
        b2 = bn.split("_")[0] if "_" in bn else bn
        if b1 == b2:
            same_brand += c
        else:
            diff_brand += c
    tot_p = same_brand + diff_brand
    print(f"\n  前 100 对混淆的错误归类：")
    print(f"    同品牌内部 : {same_brand:>5} 次 ({same_brand/max(tot_p,1)*100:.1f}%)")
    print(f"    跨品牌     : {diff_brand:>5} 次 ({diff_brand/max(tot_p,1)*100:.1f}%)")

    # ---- ⑥ 少样本类别表现 ----
    n_arr = np.array([counts.get(ids[i], 0) for i in range(n_cls)])
    print(f"\n【⑥ 少样本类别表现】（按训练样本数分档）")
    print(f"  {'档位':<16}{'类别数':>8}{'平均 Top-1':>12}")
    bins = [(0, 20), (20, 50), (50, 100), (100, 200)]
    bins_stat = []
    for lo, hi in bins:
        m = (n_arr >= lo) & (n_arr < hi)
        if m.sum() == 0:
            continue
        a = float(per_class[m].mean())
        bins_stat.append((f"{lo}-{hi} 张", int(m.sum()), a))
        print(f"  {f'{lo}-{hi} 张':<16}{m.sum():>8}{a:>12.4f}")

    # ---------------- 出图 ----------------
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])

    ax = fig.add_subplot(gs[0, 0])
    ax.bar([r[0] for r in rows_big], [r[3] for r in rows_big], color="#2980b9")
    ax.axhline(acc1, color="r", ls="--", label=f"细粒度 Top-1 = {acc1:.4f}")
    ax.set_ylabel("大类准确率"); ax.set_title("② 分层评估：按车辆大类")
    ax.legend(fontsize=8); ax.tick_params(axis="x", rotation=20)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(cm_big_pct, cmap="Blues")
    ax.set_xticks(range(len(labels_big))); ax.set_xticklabels(labels_big, rotation=45, ha="right")
    ax.set_yticks(range(len(labels_big))); ax.set_yticklabels(labels_big)
    for a in range(len(labels_big)):
        for b in range(len(labels_big)):
            if cm_big_pct[a, b] > 0.05:
                ax.text(b, a, f"{cm_big_pct[a,b]:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if cm_big_pct[a, b] > 50 else "black")
    ax.set_xlabel("预测大类"); ax.set_ylabel("真实大类")
    ax.set_title(f"③ 层次混淆矩阵（行归一化 %）\n大类准确率 {big_acc:.4f}")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 2])
    ax.hist(per_class, bins=40, color="#27ae60")
    ax.axvline(acc1, color="r", ls="--", label=f"总体 {acc1:.4f}")
    ax.set_xlabel("每类 Top-1 准确率"); ax.set_ylabel("类别数")
    ax.set_title("④ 每类准确率分布"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, :2])
    sub = np.log1p(cm)
    im = ax.imshow(sub, cmap="magma", aspect="auto")
    ax.set_xlabel("预测类别索引"); ax.set_ylabel("真实类别索引")
    ax.set_title(f"细粒度混淆矩阵（1000×1000, log(1+计数)）· Top-1 = {acc1:.4f}")
    fig.colorbar(im, ax=ax, fraction=0.02)

    ax = fig.add_subplot(gs[1, 2])
    top = pairs[:10][::-1]
    ax.barh([f"{an[:16]}\n→ {bn[:16]}" for _, _, _, an, bn in top],
            [c for c, *_ in top], color="#c0392b")
    ax.set_xlabel("错误次数"); ax.set_title("⑤ 最易混淆的 10 对")
    ax.tick_params(labelsize=6)

    fig.suptitle(f"Car-1000 评估 · {args.backbone} · 1000 类 · {args.split}集 "
                 f"Top-1={acc1:.4f} / Top-5={acc5:.4f} / 大类={big_acc:.4f}", fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS / "eval_hierarchy.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 存盘 ----------------
    out = dict(split=args.split, backbone=args.backbone, n_classes=n_cls,
               n_images=len(y), top1=round(acc1, 4), top5=round(acc5, 4),
               random_baseline=round(1 / n_cls, 5),
               big_type_acc=round(big_acc, 4),
               per_big_type=[{"type": b, "n": n, "n_classes": c, "top1": round(a, 4)}
                             for b, n, c, a in rows_big],
               big_confusion=dict(labels=labels_big, counts=cm_big.tolist()),
               per_class_min=round(float(per_class.min()), 4),
               per_class_median=round(float(np.median(per_class)), 4),
               per_class_max=round(float(per_class.max()), 4),
               confusion_same_brand=same_brand, confusion_diff_brand=diff_brand,
               few_shot=[{"bin": b, "n_classes": n, "mean_top1": round(a, 4)}
                         for b, n, a in bins_stat],
               worst_classes=[{"id": ids[i], "name": cn.get(ids[i], ""),
                               "acc": round(float(per_class[i]), 4),
                               "n_train": int(counts.get(ids[i], 0))} for i in order[:10]],
               top_confusions=[{"count": c, "true": an, "pred": bn} for c, _, _, an, bn in pairs[:20]])
    (RESULTS / "eval_car1000.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                               encoding="utf-8")

    with open(RESULTS / "eval_per_class.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "name_cn", "big_type", "n_train", "n_test", "top1_acc"])
        for i in range(n_cls):
            w.writerow([ids[i], cn.get(ids[i], ""), big.get(ids[i], ""),
                        counts.get(ids[i], 0), int(cm[i].sum()), f"{per_class[i]:.4f}"])

    with open(RESULTS / "eval_confusion_pairs.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["错误次数", "真实类别", "被误认成", "是否同品牌"])
        for c, ai, bi, an, bn in pairs[:100]:
            b1 = an.split("_")[0] if "_" in an else an
            b2 = bn.split("_")[0] if "_" in bn else bn
            w.writerow([c, an, bn, "是" if b1 == b2 else "否"])

    print(f"\n[输出] {RESULTS/'eval_hierarchy.png'}")
    print(f"[输出] {RESULTS/'eval_car1000.json'}")
    print(f"[输出] {RESULTS/'eval_per_class.csv'}")
    print(f"[输出] {RESULTS/'eval_confusion_pairs.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
