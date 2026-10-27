#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Car-1000 数据准备：原始结构 → ImageFolder 结构（train / val / test）

原始结构：
    Car-1000-dataset/
      all_images/<类别ID>/<类别ID>_<序号>.jpg      # 1000 个类别目录
      cls_info/class_info.json                     # 类别元数据
      dataset_split/dataset_split.json             # 官方 60/20/20 划分

输出结构（**原地构建，用硬链接**，不额外占空间）：
    Car-1000-dataset/prepared/
      train/<类别ID>/xxx.jpg
      val/<类别ID>/xxx.jpg
      test/<类别ID>/xxx.jpg
      class_names.txt          # 类别ID -> 中文品牌_车型
      class_taxonomy.csv       # 类别ID -> 大类 / 细分类（用于分层评估）

⚠️ 关键：类别顺序以 **all_images 目录顺序（0000→0999 升序）** 为准，
   这与 ImageFolder 的排序一致，避免主线 A 踩过的"类别索引错位"坑。

用法：
    python prepare_car1000.py                     # 原地构建（硬链接）
    python prepare_car1000.py --copy              # 改为复制（占更多空间）
    python prepare_car1000.py --check             # 只做检查不写入
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DEFAULT_ROOT = Path(r"data/Car-1000-dataset")


def load_meta(root: Path):
    info = json.loads((root / "cls_info" / "class_info.json").read_text(encoding="utf-8"))
    split = json.loads((root / "dataset_split" / "dataset_split.json").read_text(encoding="utf-8"))
    return info, split


def link_or_copy(src: Path, dst: Path, use_copy: bool) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if use_copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.link(src, dst)          # 硬链接：不额外占空间
        except OSError:
            shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--copy", action="store_true", help="复制而非硬链接")
    ap.add_argument("--check", action="store_true", help="只检查不写入")
    args = ap.parse_args()

    root: Path = args.root
    out: Path = args.out or (root / "prepared")
    if not root.exists():
        raise SystemExit(f"[FAIL] 数据集不存在: {root}")

    print("=" * 92)
    print(f"Car-1000 数据准备 | 源 {root}")
    print(f"                   输出 {out} | 方式 {'复制' if args.copy else '硬链接'}")
    print("=" * 92)

    info, split = load_meta(root)
    meta = {i["id"]: i for i in info}

    # ---- 类别顺序：以 all_images 目录升序为准（与 ImageFolder 一致）----
    img_root = root / "all_images"
    class_ids = sorted(d.name for d in img_root.iterdir() if d.is_dir())
    print(f"[类别] all_images 目录数 {len(class_ids)}  范围 {class_ids[0]}..{class_ids[-1]}")

    ids_json = sorted(i["id"] for i in info)
    if class_ids != ids_json:
        only_dir = set(class_ids) - set(ids_json)
        only_json = set(ids_json) - set(class_ids)
        print(f"[类别] ⚠️ 目录与 JSON 的 ID 集合不一致：仅在目录 {len(only_dir)} 个，"
              f"仅在 JSON {len(only_json)} 个 → **以目录顺序为准**")
    else:
        print("[类别] ✅ 目录顺序与 JSON 的 ID 集合一致")

    miss_meta = [c for c in class_ids if c not in meta]
    if miss_meta:
        print(f"[类别] ⚠️ {len(miss_meta)} 个目录在 class_info.json 中无元数据")

    # ---- 统计各 split ----
    tot = Counter()
    per_cls = []
    for rec in split:
        cid = rec["id"]
        counts = {k: len(rec.get(f"{k}_data") or []) for k in ("train", "val", "test")}
        per_cls.append((cid, counts))
        for k, v in counts.items():
            tot[k] += v
    print(f"\n[划分] train {tot['train']} | val {tot['val']} | test {tot['test']} "
          f"| 合计 {sum(tot.values())}")

    if args.check:
        print("\n[check] 仅检查，未写入任何文件")
        return 0

    # ---- 构建 ----
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    written = Counter()
    missing = Counter()
    for idx, rec in enumerate(split, 1):
        cid = rec["id"]
        src_dir = img_root / cid
        for spl in ("train", "val", "test"):
            names = rec.get(f"{spl}_data") or []
            if not names:
                continue
            dst_dir = out / spl / cid
            for name in names:
                src = src_dir / name
                if not src.exists():
                    missing[spl] += 1
                    continue
                link_or_copy(src, dst_dir / name, args.copy)
                written[spl] += 1
        if idx % 200 == 0:
            el = time.time() - t0
            print(f"  进度 {idx}/{len(split)} 类 | 已链接 {sum(written.values())} 张 "
                  f"| 用时 {el/60:.1f} 分钟", flush=True)

    print(f"\n[完成] 用时 {(time.time()-t0)/60:.1f} 分钟")
    for spl in ("train", "val", "test"):
        print(f"  {spl:<6} {written[spl]:>7} 张" + (f"  (缺失 {missing[spl]} 张)" if missing[spl] else ""))

    # ---- 元数据（顺序必须与目录一致）----
    with open(out / "class_names.txt", "w", encoding="utf-8") as f:
        f.write("class_id\tname_cn\tname_en\tsplit_of\n")
        for cid in class_ids:
            m = meta.get(cid, {})
            cn = m.get("name_from_new", "")
            old = m.get("name_from_old", "")
            en = old.split("==")[1] if "==" in old else ""
            f.write(f"{cid}\t{cn}\t{en}\t\n")

    with open(out / "class_taxonomy.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "name_cn", "big_type", "sub_type", "big_type_en", "sub_type_en",
                    "n_train", "n_val", "n_test"])
        cnt_map = {c: d for c, d in per_cls}
        for cid in class_ids:
            m = meta.get(cid, {})
            ti = m.get("type_info", "")
            parts = ti.split("==")
            cn_part = parts[0].split("####") if parts else []
            en_part = parts[1].split("####") if len(parts) > 1 else []
            c = cnt_map.get(cid, {})
            w.writerow([cid, m.get("name_from_new", ""),
                        cn_part[0] if cn_part else "", cn_part[1] if len(cn_part) > 1 else "",
                        en_part[0] if en_part else "", en_part[1] if len(en_part) > 1 else "",
                        c.get("train", 0), c.get("val", 0), c.get("test", 0)])

    print(f"\n[输出] {out / 'class_names.txt'}")
    print(f"[输出] {out / 'class_taxonomy.csv'}")
    print(f"\n[类别顺序] 前 3: {class_ids[:3]}  后 3: {class_ids[-3:]}")
    print(f"[目录结构] {out}\\<train|val|test>\\<类别ID>\\*.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
