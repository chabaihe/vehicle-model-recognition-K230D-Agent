#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Car-1000 计时与显存实验 —— **正式训练前必做**

目的：在 83,756 张训练图上，实测出
      ① 每个 epoch 的真实耗时（决定总 epoch 数与计划可行性）
      ② 峰值显存（确认 8 GB 够用）
      ③ 数据加载是否成为瓶颈（GPU 利用率 / worker 数影响）

做法：不训练完整 epoch，而是**跑固定数量的 batch** 外推，
      同时测出纯数据加载速度与纯前向反向速度，定位瓶颈。

用法：
    python bench_car1000.py                                  # 默认配置
    python bench_car1000.py --batches 60 --batch 64 --workers 8
    python bench_car1000.py --backbone resnet50 --size 224
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, resnet18, mobilenet_v3_large, efficientnet_b0

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--backbone", default="resnet50", choices=list(BUILDERS))
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batches", type=int, default=50, help="实测的 batch 数（用于外推）")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 96)
    print(f"Car-1000 计时实验 | {args.backbone} | {args.size}px | batch {args.batch} | workers {args.workers}")
    print(f"设备: {torch.cuda.get_device_name(0) if dev.type=='cuda' else 'CPU'} | 类别数 {N_CLASSES}")
    print("=" * 96)

    train_dir = args.data / "train"
    if not train_dir.exists():
        raise SystemExit(f"[FAIL] 训练目录不存在: {train_dir}")
    n_cls = len([d for d in train_dir.iterdir() if d.is_dir()])
    print(f"[数据] {train_dir}  |  类别 {n_cls} 个")

    tf = transforms.Compose([
        transforms.Resize((args.size + 32, args.size + 32)),
        transforms.RandomCrop((args.size, args.size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    t0 = time.time()
    ds = datasets.ImageFolder(str(train_dir), tf)
    print(f"[数据] 训练集 {len(ds)} 张（索引构建 {time.time()-t0:.1f}s）")

    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True,
                    persistent_workers=args.workers > 0,
                    prefetch_factor=4 if args.workers > 0 else None)

    # ---------- ① 纯数据加载速度 ----------
    # ⚠️ 必须跳过前若干个 batch：DataLoader 的 worker 冷启动（打开 8 个进程、
    #    建立 1000 个类别目录的索引）会严重低估稳态速度。
    print("\n[1/3] 测纯数据加载速度（跳过 worker 冷启动）...")
    WARMUP = 8
    t0 = time.time(); nb = 0
    for x, y in dl:
        nb += 1
        if nb == WARMUP:
            t0 = time.time(); nb = 0      # 冷启动结束，重新计时
        if nb >= 40:
            break
    load_ips = nb * args.batch / (time.time() - t0)
    print(f"      数据加载 {load_ips:.1f} img/s  （稳态，{nb} 个 batch）")

    # ---------- ② 训练步（前向+反向+优化器）----------
    print(f"\n[2/3] 测训练步（{args.batches} 个 batch）...")
    model = build_model(args.backbone).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    if dev.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)

    it = iter(dl)
    # 预热
    for _ in range(3):
        x, y = next(it)
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            loss = crit(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    if dev.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time(); nb = 0; losses = []
    for _ in range(args.batches):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl); x, y = next(it)
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            loss = crit(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        losses.append(float(loss.detach())); nb += 1
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    train_ips = nb * args.batch / dt
    peak = torch.cuda.max_memory_allocated(dev) / 1024**3 if dev.type == "cuda" else 0.0
    print(f"      训练吞吐 {train_ips:.1f} img/s  |  峰值显存 {peak:.2f} GB  |  首个 loss {losses[0]:.3f}")

    # ---------- ③ GPU 利用率采样 ----------
    print("\n[3/3] 采样 GPU 利用率（判断是否 I/O 瓶颈）...")
    import subprocess
    util = []
    t0 = time.time()
    while time.time() - t0 < 8:
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,power.draw",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                u, c, p = [s.strip() for s in r.stdout.strip().split(",")]
                util.append((float(u), float(c), float(p)))
        except Exception:
            pass
        # 同时持续训练，保证有负载
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl); x, y = next(it)
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            loss = crit(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        time.sleep(0.3)

    if util:
        import statistics
        us = [u[0] for u in util]; cs = [u[1] for u in util]; ps = [u[2] for u in util]
        print(f"      GPU 利用率: 均值 {statistics.mean(us):.1f}%  最大 {max(us):.0f}%")
        print(f"      SM 频率:    均值 {statistics.mean(cs):.0f} MHz  最大 {max(cs):.0f} MHz")
        print(f"      功耗:       均值 {statistics.mean(ps):.1f} W  最大 {max(ps):.1f} W")
    else:
        print("      （无法读取 nvidia-smi）")

    # ---------- 外推 ----------
    n_train = len(ds)
    n_batches = n_train // args.batch
    sec_per_epoch = n_batches * (args.batch / train_ips)
    sec_per_epoch_io = n_train / load_ips
    bottleneck = "数据加载(I/O)" if sec_per_epoch_io > sec_per_epoch else "GPU 计算"

    print("\n" + "=" * 96)
    print("【结论】")
    print(f"  训练集规模        : {n_train} 张（{n_batches} 个 batch/epoch）")
    print(f"  训练吞吐          : {train_ips:.1f} img/s")
    print(f"  数据加载吞吐      : {load_ips:.1f} img/s")
    print(f"  **每 epoch 耗时** : {sec_per_epoch/60:.1f} 分钟（按训练吞吐外推）")
    print(f"  仅数据加载耗时    : {sec_per_epoch_io/60:.1f} 分钟")
    print(f"  **瓶颈判断**      : {bottleneck}")
    print(f"  峰值显存          : {peak:.2f} GB / 8 GB")
    print()
    for ep in (5, 10, 20, 30):
        print(f"  {ep:>2} 个 epoch 预计 : {sec_per_epoch*ep/3600:.1f} 小时")
    print("=" * 96)

    summary = dict(backbone=args.backbone, size=args.size, batch=args.batch,
                   workers=args.workers, n_train=n_train, n_classes=N_CLASSES,
                   train_img_per_sec=round(train_ips, 1), load_img_per_sec=round(load_ips, 1),
                   sec_per_epoch=round(sec_per_epoch, 1),
                   min_per_epoch=round(sec_per_epoch / 60, 2),
                   bottleneck=bottleneck, peak_vram_gb=round(peak, 3),
                   gpu_util_mean=round(sum(u[0] for u in util) / len(util), 1) if util else None)
    out = RESULTS / "bench_car1000.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[输出] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
