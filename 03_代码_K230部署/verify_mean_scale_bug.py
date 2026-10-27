# -*- coding: utf-8 -*-
r"""
验证推论：kmodel 里 mean/std 的量纲搞错了

官方文档说明：
    input_range=[0,255] 时，Dequantize 只做类型转换（uint8→float32，保持 0~255），
    随后 Normalization 用的是 **0~255 量纲**的 mean/std。

我们的转换脚本设的是：
    input_range = [0, 255]
    mean = [0.485, 0.456, 0.406]   ← 这是 **0~1 量纲**的 ImageNet 系数！
    std  = [0.229, 0.224, 0.225]

→ 在 0~255 量纲下，减 0.485 约等于「不减」，除 0.229 约等于「放大 4.4 倍」
→ 输入实际变成 ~0~1100 的巨大数值 → logits 爆炸 → softmax 饱和 → 100%

本脚本在 PC 上复现这个"错误归一化"，看是否得到与板端一致的饱和结果。

用法：
    python verify_mean_scale_bug.py --img <图片>
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
from torchvision.models import resnet50

HERE = Path(__file__).resolve().parent
TOPIC = HERE.parent
RESULTS = TOPIC / "results"
DATA = TOPIC / "data" / "Car-1000"

N_CLASSES = 1000
MEAN01 = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # 0~1 量纲
STD01 = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MEAN255 = MEAN01 * 255.0                                     # 0~255 量纲
STD255 = STD01 * 255.0


def load_labels():
    ids = sorted(x.name for x in (DATA / "train").iterdir() if x.is_dir())
    cn = {}
    with open(DATA / "class_taxonomy.csv", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            cn[r["class_id"]] = r["name_cn"]
    return ids, cn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=Path, required=True)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    ids, cn = load_labels()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    model.load_state_dict(torch.load(RESULTS / "best_resnet50.pth",
                                     map_location=dev, weights_only=True))
    model = model.to(dev).eval()

    im = Image.open(args.img).convert("RGB").resize((args.size, args.size), Image.BILINEAR)
    x0 = np.asarray(im, dtype=np.float32)          # 0~255

    cases = {
        "① 正确（PC 端做法）":
            (x0 / 255.0 - MEAN01) / STD01,
        "② 板端嫌疑：0~255 量纲下用 0~1 的 mean/std":
            (x0 - MEAN01) / STD01,
        "③ 若 mean/std 正确按 0~255 给":
            (x0 - MEAN255) / STD255,
        "④ 只减 0.485 不除 std":
            (x0 - MEAN01),
    }

    print("=" * 100)
    print(f"验证 mean/std 量纲问题 | {Path(args.img).name}")
    print("=" * 100)
    print(f"{'情况':<44}{'输入范围':>22}{'Top1':>10}   {'类别':<26}")
    print("-" * 100)
    for name, arr in cases.items():
        t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).unsqueeze(0).to(dev)
        with torch.no_grad():
            logits = model(t)[0].cpu().numpy()
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        i = int(p.argmax())
        rng = f"[{arr.min():7.1f},{arr.max():7.1f}]"
        flag = "  ← 饱和" if p[i] > 0.90 else ""
        print(f"{name:<44}{rng:>22}{p[i]*100:>9.2f}%   {cn.get(ids[i], ids[i]):<26}{flag}")

    print("\n" + "=" * 100)
    print("【判读】")
    print("  板端实际: Top1 = 100.0%  日产_Note EV")
    print("  若 ② 出现 ~100% 饱和  → 证实是 mean/std 量纲错误")
    print("  修正方法：input_range 改 [0,1]（让反量化直接输出 0~1），mean/std 保持 ImageNet 值")
    print("            或者 input_range 保持 [0,255]，但 mean/std 乘 255")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
