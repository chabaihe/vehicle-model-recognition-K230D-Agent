#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Car-1000 部署脚本（阶段 1）：导出 ONNX + 对拍验证

在**当前训练环境（Python 3.14 + torch）**里运行：
  ① 导出 cars1000.onnx（静态 batch=1，适合 nncase）
  ② **对拍验证**：同一批测试图，PyTorch vs ONNX 的预测是否一致
     ⚠️ 这一步必须通过，再进入 nncase 转换 —— 否则会误判为 nncase 的问题

产出（deploy/）：
  cars1000_resnet50_224.onnx      部署用模型
  class_names.txt                 类别表（顺序与模型输出索引严格对应）
  class_names_en.txt              英文版类别表（板端若无法显示中文可用）
  verify_report.json              对拍结果

用法：
    python export_onnx_car1000.py
    python export_onnx_car1000.py --n 600        # 对拍样本数
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet50, resnet18, mobilenet_v3_large, efficientnet_b0

HERE = Path(__file__).resolve().parent
TOPIC = HERE.parent
RESULTS = TOPIC / "results"
DEPLOY = TOPIC / "deploy"
DEPLOY.mkdir(parents=True, exist_ok=True)
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


def write_class_files(ids: list[str], cn: dict, en: dict, big: dict) -> None:
    (DEPLOY / "class_names.txt").write_text("\n".join(ids), encoding="utf-8")
    with open(DEPLOY / "class_names_full.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "class_id", "name_cn", "name_en", "big_type"])
        for i, cid in enumerate(ids):
            w.writerow([i, cid, cn.get(cid, ""), en.get(cid, ""), big.get(cid, "")])
    (DEPLOY / "class_names_en.txt").write_text(
        "\n".join(en.get(c, c) for c in ids), encoding="utf-8")
    (DEPLOY / "class_names_cn.txt").write_text(
        "\n".join(cn.get(c, c) for c in ids), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50", choices=list(BUILDERS))
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--n", type=int, default=800, help="对拍验证样本数")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    ckpt = args.ckpt or (RESULTS / f"best_{args.backbone}.pth")
    if not ckpt.exists():
        raise SystemExit(f"[FAIL] 权重不存在: {ckpt}")

    print("=" * 96)
    print(f"Car-1000 ONNX 导出与对拍 | {args.backbone} | {args.size}px | opset {args.opset}")
    print("=" * 96)

    # ---- 类别表（顺序以 train 目录为准）----
    ids = sorted(x.name for x in (DATA / "train").iterdir() if x.is_dir())
    cn, en, big = {}, {}, {}
    tax = DATA / "class_taxonomy.csv"
    if tax.exists():
        with open(tax, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                cn[r["class_id"]] = r["name_cn"]
                big[r["class_id"]] = r["big_type"]
    # 英文名从原始 class_info.json 取
    raw = Path(r"data/Car-1000-dataset/cls_info/class_info.json")
    if raw.exists():
        for item in json.loads(raw.read_text(encoding="utf-8")):
            old = item.get("name_from_old", "")
            en[item["id"]] = old.split("==")[1] if "==" in old else item.get("name_from_new", "")
    print(f"[类别] {len(ids)} 类（顺序取自 train 目录）| 中文名 {len(cn)} | 英文名 {len(en)}")

    model = build_model(args.backbone)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    print(f"[模型] 参数 {sum(p.numel() for p in model.parameters())/1e6:.2f} M | {ckpt.name}")

    out = DEPLOY / f"cars1000_{args.backbone}_{args.size}.onnx"
    dummy = torch.randn(1, 3, args.size, args.size)
    with torch.no_grad():
        torch.onnx.export(model, dummy, str(out), input_names=["input"], output_names=["logits"],
                          opset_version=args.opset, dynamo=False)
    try:
        import onnx
        onnx.checker.check_model(onnx.load(str(out)))
        print("[校验] ONNX checker 通过")
    except ImportError:
        print("[校验] 未安装 onnx，跳过")
    print(f"[输出] {out}  （{out.stat().st_size/1024**2:.2f} MB）")
    write_class_files(ids, cn, en, big)
    print(f"[输出] {DEPLOY/'class_names.txt'}（类别ID顺序）")
    print(f"[输出] {DEPLOY/'class_names_cn.txt'}（中文名，板端显示用）")
    print(f"[输出] {DEPLOY/'class_names_en.txt'}（英文名，板端无中文字体时用）")
    print(f"[输出] {DEPLOY/'class_names_full.csv'}（完整映射表）")

    if args.skip_verify:
        return 0

    # ---------------- 对拍验证 ----------------
    print("\n" + "=" * 96)
    print(f"【对拍验证】PyTorch vs ONNX（{args.n} 张测试图）")
    print("=" * 96)
    try:
        import onnxruntime as ort
    except ImportError:
        print("[跳过] 未安装 onnxruntime")
        return 0

    tf = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    ds = datasets.ImageFolder(str(DATA / "test"), tf)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(ds), min(args.n, len(ds)), replace=False)

    xs = np.stack([ds[i][0].numpy() for i in sel])
    ys = np.array([ds[i][1] for i in sel])
    print(f"       样本 {len(sel)} 张 | 覆盖类别 {len(set(ys.tolist()))}")

    # PyTorch
    pt = []
    with torch.no_grad():
        for i in range(0, len(xs), 32):
            pt.append(model(torch.from_numpy(xs[i:i+32])).numpy())
    pt = np.concatenate(pt)

    # ONNX（ASCII 临时目录，规避中文路径问题）
    wd = Path(tempfile.gettempdir()) / "car1000_verify"
    wd.mkdir(parents=True, exist_ok=True)
    onnx_ascii = wd / out.name
    shutil.copy2(out, onnx_ascii)
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(str(onnx_ascii), so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    t0 = time.time()
    ort_out = np.zeros_like(pt)
    for i in range(len(xs)):                      # 静态 batch=1，逐张推理
        ort_out[i] = sess.run(None, {inp: xs[i:i+1]})[0][0]
    lat = (time.time() - t0) / len(xs) * 1000

    agree = float((pt.argmax(1) == ort_out.argmax(1)).mean())
    pt1 = float((pt.argmax(1) == ys).mean())
    on1 = float((ort_out.argmax(1) == ys).mean())
    maxdiff = float(np.abs(pt - ort_out).max())

    print(f"\n  PyTorch Top-1 : {pt1:.4f}")
    print(f"  ONNX    Top-1 : {on1:.4f}")
    print(f"  **预测一致率** : {agree*100:.2f}%")
    print(f"  logits 最大差异: {maxdiff:.6f}")
    print(f"  ONNX CPU 单张延迟: {lat:.1f} ms")

    ok = agree > 0.995 and abs(pt1 - on1) < 0.01
    print("\n" + ("[✅ 通过] 可以进入 nncase 转 kmodel 环节。"
                  if ok else
                  "[❌ 未通过] 先查这里，不要动 nncase：\n"
                  "   1) 导出时是否 model.eval()？（BatchNorm 统计量）\n"
                  "   2) 归一化 mean/std 是否与训练一致？\n"
                  "   3) ONNX 里是否重复加了预处理？"))
    (DEPLOY / "verify_report.json").write_text(json.dumps(dict(
        backbone=args.backbone, size=args.size, n=len(sel),
        pt_top1=round(pt1, 4), onnx_top1=round(on1, 4), agree=round(agree, 4),
        logits_maxdiff=round(maxdiff, 6), onnx_cpu_latency_ms=round(lat, 1),
        passed=bool(ok)), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[输出] {DEPLOY/'verify_report.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
