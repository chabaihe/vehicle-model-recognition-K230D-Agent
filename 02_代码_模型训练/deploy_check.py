#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
K230 部署可行性实测：ONNX INT8 量化后精度损失

背景：K230 KPU 只支持 INT8/INT16 推理。所以"模型能不能部署"的真正门槛不是
      显存、也不是算力，而是 **INT8 量化后精度掉多少**。
      本脚本在 PC 上用 onnxruntime 的 INT8 量化做**等价模拟**，提前量化风险。

产出：
  - 各骨干 FP32(ONNX) vs INT8 的 Top-1 / Top-5 对比
  - 精度损失（百分点）
  - 模型体积、CPU 推理延迟（作为相对速度参考）

注意：
  - onnxruntime 的动态量化与 nncase 的 PTQ 不完全相同，但**量级和趋势可参考**；
  - 要拿到 K230 上的真实数字，仍需在板端跑一遍（本脚本用于提前判断风险）。

用法：
    python deploy_check.py --data data/prepared --n 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

HERE = Path(__file__).resolve().parent
TOPIC = HERE.parent
RESULTS = TOPIC / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

N_CLASSES = 196
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_eval_set(root: Path, n: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    """从 ImageFolder 结构的 test/ 里取 n 张已预处理的图片 (N,3,H,W) float32。"""
    from PIL import Image
    test_dir = root / "test"
    if not test_dir.exists():
        raise SystemExit(f"[FAIL] 找不到 {test_dir}")
    classes = sorted(d.name for d in test_dir.iterdir() if d.is_dir())
    cls2idx = {c: i for i, c in enumerate(classes)}
    files = [(p, cls2idx[p.parent.name]) for p in test_dir.rglob("*")
             if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(files), min(n, len(files)), replace=False)
    xs, ys = [], []
    for i in idx:
        p, y = files[i]
        im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
        a = (a - IMAGENET_MEAN) / IMAGENET_STD
        xs.append(a.transpose(2, 0, 1))
        ys.append(y)
    return np.stack(xs), np.array(ys)


def evaluate_onnx(sess, x: np.ndarray, y: np.ndarray, batch: int = 32) -> tuple[float, float, float]:
    """返回 (top1, top5, 单张延迟 ms)。"""
    c1 = c5 = tot = 0
    inp = sess.get_inputs()[0].name
    t0 = time.time()
    for i in range(0, len(x), batch):
        xb = x[i:i + batch]
        out = sess.run(None, {inp: xb})[0]
        yb = y[i:i + batch]
        c1 += (out.argmax(1) == yb).sum()
        k = min(5, out.shape[1])
        c5 += (np.argsort(-out, axis=1)[:, :k] == yb[:, None]).any(1).sum()
        tot += len(yb)
    dt = (time.time() - t0) / tot * 1000
    return c1 / tot, c5 / tot, dt


def export_dynamic_batch(ckpt: Path, backbone: str, size: int, out: Path) -> None:
    """从 PyTorch 权重导出**动态 batch** 的 ONNX（否则只能单张推理）。"""
    import torch
    import torch.nn as nn
    from torchvision.models import (resnet50, resnet18, mobilenet_v3_large,
                                    efficientnet_b0)

    builders = {"resnet50": resnet50, "resnet18": resnet18,
                "mobilenet_v3_large": mobilenet_v3_large,
                "efficientnet_b0": efficientnet_b0}
    m = builders[backbone](weights=None)
    if hasattr(m, "fc"):
        m.fc = nn.Linear(m.fc.in_features, N_CLASSES)
    elif hasattr(m, "classifier"):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, N_CLASSES)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    m.eval()
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(m, dummy, str(out), input_names=["input"], output_names=["logits"],
                      opset_version=17, dynamo=False,
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(r"data/prepared"))
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--n", type=int, default=1200, help="评估图片数（越大越准，越慢）")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    try:
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        raise SystemExit("需要 onnxruntime: python -m pip install onnxruntime")

    print("=" * 100)
    print(f"K230 部署可行性实测 · INT8 量化精度损失 · 输入 {args.size}px · 评估 {args.n} 张")
    print("=" * 100)

    x, y = load_eval_set(args.data, args.n, args.size)
    print(f"[数据] 评估集 {x.shape}, 标签 {y.shape}, 覆盖类别 {len(np.unique(y))}")

    workdir = Path(tempfile.gettempdir()) / "k230_quant_bench"
    workdir.mkdir(parents=True, exist_ok=True)

    # 优先用已保存的 PyTorch 权重重新导出（带动态 batch）；否则退回现成 ONNX
    ckpts = sorted(RESULTS.glob("best_*.pth"))
    jobs = []
    for ck in ckpts:
        backbone = ck.stem.replace("best_", "")
        jobs.append((backbone, ck))
    if not jobs:
        print("[警告] 未找到 best_*.pth，退回使用现成 ONNX（仅支持 batch=1）")

    rows = []
    print("\n" + "=" * 100)
    print(f"{'模型':<20}{'FP32 Top-1':>12}{'INT8 Top-1':>12}{'损失(pp)':>10}"
          f"{'FP32 Top-5':>12}{'INT8 Top-5':>12}{'体积MB':>10}{'ms/张':>9}")
    print("=" * 100)

    for backbone, ck in jobs:
        fp32_path = workdir / f"{backbone}_dyn_fp32.onnx"
        int8_path = workdir / f"{backbone}_dyn_int8.onnx"
        if not fp32_path.exists():
            try:
                export_dynamic_batch(ck, backbone, args.size, fp32_path)
            except Exception as e:
                print(f"  {backbone}: 导出失败 {str(e)[:70]}")
                continue
        try:
            quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8)
        except Exception as e:
            print(f"  {backbone}: 量化失败 {str(e)[:70]}")
            continue

        so = ort.SessionOptions()
        so.log_severity_level = 3
        s_fp32 = ort.InferenceSession(str(fp32_path), so, providers=["CPUExecutionProvider"])
        s_int8 = ort.InferenceSession(str(int8_path), so, providers=["CPUExecutionProvider"])

        t1f, t5f, _ = evaluate_onnx(s_fp32, x, y, batch=args.batch)
        t1q, t5q, msq = evaluate_onnx(s_int8, x, y, batch=args.batch)

        mb_f = fp32_path.stat().st_size / 1024**2
        mb_q = int8_path.stat().st_size / 1024**2
        rows.append(dict(model=backbone, fp32_top1=round(float(t1f), 4),
                         int8_top1=round(float(t1q), 4),
                         drop_pp=round(float(t1f - t1q) * 100, 2),
                         fp32_top5=round(float(t5f), 4), int8_top5=round(float(t5q), 4),
                         size_fp32_mb=round(mb_f, 2), size_int8_mb=round(mb_q, 2),
                         latency_ms=round(msq, 1)))
        print(f"{backbone:<20}{t1f:>12.4f}{t1q:>12.4f}{(t1f-t1q)*100:>10.2f}"
              f"{t5f:>12.4f}{t5q:>12.4f}{mb_q:>10.2f}{msq:>9.1f}")
    print("=" * 100)

    if not rows:
        return 1

    best = max(rows, key=lambda r: r["int8_top1"])
    print(f"\n[结论] INT8 量化后精度最高: {best['model']} → Top-1 {best['int8_top1']:.4f}"
          f"（FP32 {best['fp32_top1']:.4f}，损失 {best['drop_pp']:.2f} pp）")

    # K230D 的 128MB DDR 预算对照
    print("\n【K230D 内存预算对照】（官方参数：DDR 128MB LPDDR4）")
    for r in rows:
        need = r["size_int8_mb"]
        print(f"  {r['model']:<22} INT8 模型 {need:>6.2f} MB  = 128MB 的 {need/128*100:>5.1f}%"
              f"  → 剩余约 {128-need:>6.1f} MB 给运行时与激活缓冲")
    print("  注：实际还需预留 RTOS/CanMV 运行时、图像缓冲、激活中间张量；"
          "模型建议不超过总内存的 ~30%。")

    with open(RESULTS / "test8_K230量化评估.json", "w", encoding="utf-8") as f:
        json.dump(dict(input_size=args.size, n_eval=len(x), results=rows), f,
                  ensure_ascii=False, indent=2)
    print(f"\n[输出] {RESULTS / 'test8_K230量化评估.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
