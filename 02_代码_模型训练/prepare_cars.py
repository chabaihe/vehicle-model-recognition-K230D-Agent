#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Stanford Cars 数据准备：把原始平铺格式整理成 ImageFolder 结构

原始格式（本仓库下载到的版本）：
    stanford_cars/
      cars_train/                     # 8144 张，平铺，无类别子目录
      cars_test/                      # 8041 张
      devkit/cars_meta.mat            # 196 个类名
      devkit/cars_train_annos.mat     # 训练标注（bbox + 类别）
      cars_test_annos_withlabels.mat  # 测试标注（含类别，官方版本通常没有）

整理后：
    prepared/
      train/<类名>/xxx.jpg            # 默认用**硬链接**，几乎不额外占磁盘
      test/<类名>/xxx.jpg
      class_names.txt
      class_distribution.csv

用法：
    python prepare_cars.py --src data/stanford_cars --out data/prepared
    python prepare_cars.py --copy      # 不用硬链接，改为复制（占更多空间）
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np


def read_mat(path: Path):
    try:
        from scipy.io import loadmat
    except ImportError:
        raise SystemExit("需要 scipy: python -m pip install scipy")
    return loadmat(str(path))


def get_class_names(meta_path: Path) -> list[str]:
    m = read_mat(meta_path)
    raw = m["class_names"][0]
    names = []
    for item in raw:
        v = item[0] if isinstance(item, np.ndarray) else item
        names.append(str(v))
    return names


def parse_annos(anno_path: Path, key: str) -> list[tuple[str, int]]:
    """返回 [(文件名, 类别1-based)]。"""
    m = read_mat(anno_path)
    annos = m[key][0]
    out = []
    for a in annos:
        fname = str(a["fname"][0])
        cls = int(a["class"][0][0])
        out.append((fname, cls))
    return out


def link_or_copy(src: Path, dst: Path, use_copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if use_copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.link(src, dst)          # 硬链接：同一文件多路径，不额外占空间
        except OSError:
            shutil.copy2(src, dst)     # 跨卷等情况退回复制


def safe_dirname(name: str, used: set[str]) -> str:
    """
    生成合法的目录名。

    关键坑：Stanford Cars 里有类名含斜杠，例如
      'AM General Hummer SUV 2000/2002'
    若只把 '/' 换成 '_'，虽然能得到合法目录名，但会让人以为类名本身带下划线；
    更严重的是**必须替换掉所有路径分隔符**，否则会被写成嵌套目录，
    导致 ImageFolder 把子目录当成一个新类别（196 类变 197 类）。
    """
    bad = '/\\:*?"<>|'
    out = "".join(("_" if ch in bad else ch) for ch in name).strip().strip(".")
    out = " ".join(out.split())                     # 折叠多余空白
    if not out:
        out = "class"
    base, i = out, 2
    while out in used:                              # 保证唯一
        out = f"{base}_{i}"; i += 1
    used.add(out)
    return out


def build(split: str, img_dir: Path, annos: list[tuple[str, int]],
          dirnames: list[str], out_root: Path, use_copy: bool) -> Counter:
    cnt: Counter = Counter()
    missing = 0
    for fname, cls in annos:
        if cls < 1 or cls > len(dirnames):
            continue
        cname = dirnames[cls - 1]
        src = img_dir / fname
        if not src.exists():
            missing += 1
            continue
        link_or_copy(src, out_root / split / cname / fname, use_copy)
        cnt[cname] += 1
    if missing:
        print(f"  [警告] {split}: {missing} 张图片未找到")
    return cnt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path(r"data/stanford_cars"))
    ap.add_argument("--out", type=Path, default=Path(r"data/prepared"))
    ap.add_argument("--copy", action="store_true", help="复制而非硬链接")
    args = ap.parse_args()

    src, out = args.src, args.out
    if not src.exists():
        raise SystemExit(f"[FAIL] 源目录不存在: {src}")

    meta = src / "devkit" / "cars_meta.mat"
    tr_anno = src / "devkit" / "cars_train_annos.mat"
    # 含标签的测试标注可能在两处
    te_cands = [src / "cars_test_annos_withlabels.mat",
                src / "devkit" / "cars_test_annos_withlabels.mat"]

    print("[1/4] 读取类别名...")
    class_names = get_class_names(meta)
    print(f"      共 {len(class_names)} 类；示例: {class_names[:3]} ...")

    print("[2/4] 读取标注...")
    tr_list = parse_annos(tr_anno, "annotations")
    print(f"      训练标注 {len(tr_list)} 条")
    te_list = None
    for c in te_cands:
        if c.exists():
            m = read_mat(c)
            key = [k for k in m.keys() if not k.startswith("__")][0]
            te_list = parse_annos(c, key)
            print(f"      测试标注 {len(te_list)} 条（含标签，来自 {c.name}）")
            break
    if te_list is None:
        print("      [警告] 未找到含标签的测试标注 → 将只整理训练集")

    print(f"[3/4] 构建 ImageFolder 结构 → {out}  ({'复制' if args.copy else '硬链接'})")
    # 先清理旧结果，避免上一次的错误嵌套目录残留
    for split in ("train", "test"):
        d = out / split
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    used: set[str] = set()
    dirnames = [safe_dirname(n, used) for n in class_names]
    renamed = [(a, b) for a, b in zip(class_names, dirnames) if a != b]
    if renamed:
        print(f"      {len(renamed)} 个类名含非法字符，已规范化。示例:")
        for a, b in renamed[:3]:
            print(f"        {a!r} → {b!r}")

    tr_cnt = build("train", src / "cars_train", tr_list, dirnames, out, args.copy)
    print(f"      train: {sum(tr_cnt.values())} 张 / {len(tr_cnt)} 类")
    te_cnt = Counter()
    if te_list:
        te_cnt = build("test", src / "cars_test", te_list, dirnames, out, args.copy)
        print(f"      test : {sum(te_cnt.values())} 张 / {len(te_cnt)} 类")
    if len(tr_cnt) != len(class_names):
        print(f"      [警告] 实际类别数 {len(tr_cnt)} ≠ 类别表 {len(class_names)}")

    print("[4/4] 写出元数据...")
    (out / "class_names.txt").write_text(
        "\n".join(f"{i+1}\t{orig}\t{d}" for i, (orig, d) in enumerate(zip(class_names, dirnames))),
        encoding="utf-8")
    with open(out / "class_distribution.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["类别目录名", "原始类名", "训练样本数", "测试样本数"])
        for orig, d in zip(class_names, dirnames):
            w.writerow([d, orig, tr_cnt.get(d, 0), te_cnt.get(d, 0)])

    per = np.array([tr_cnt.get(d, 0) for d in dirnames])
    print(f"\n[统计] 训练集每类样本: 最少 {per.min()} / 最多 {per.max()} / 中位 {int(np.median(per))} / 总 {per.sum()}")
    print(f"       不均衡比（最多/最少）= {per.max()/max(per.min(),1):.2f} / 完全缺失类别数 {(per == 0).sum()}")
    if len(te_cnt):
        per_te = np.array([te_cnt.get(d, 0) for d in dirnames])
        print(f"[统计] 测试集每类样本: 最少 {per_te.min()} / 最多 {per_te.max()} / 中位 {int(np.median(per_te))}")
    print(f"\n[完成] 输出目录: {out}\n       类别表: {out / 'class_names.txt'}\n       分布表: {out / 'class_distribution.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
