#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
混淆结构分析（改进版）：把"哪些车型互相认错"提升为"为什么会认错"

判定层级（从最相似到最不同）：
  ① 同车系·仅年份不同   —— 同款车不同年款（如 Bentley Continental GT 2012 → 2007）
  ② 同车系·车身形式不同 —— 同款车不同车身（如 Express Van → Express Cargo Van）
  ③ 同平台·跨品牌       —— 共享平台的换标车（如 Dodge Sprinter → Mercedes-Benz Sprinter）
  ④ 不同车型            —— 真正的跨车型误认

做法：先从类名剥离"年份 + 车身形式词 + 座位/车门配置"，得到**车型血脉（lineage）**，
      再按层级判定。这样能避免把"同款车的不同车身"误判为"不同车型"。

用法：
    python analyze_confusion.py                # 自动选 results 下最新的混淆对 CSV
    python analyze_confusion.py --csv <path> --top 100
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

# 车身形式 / 配置词（剥离后用于比较车型血脉）
BODY_WORDS = [
    "crew cab", "regular cab", "extended cab", "quad cab", "super cab", "king cab",
    "cargo van", "passenger van", "minivan", "van", "wagon", "hatchback", "sedan",
    "coupe", "convertible", "suv", "pickup", "roadster", "cabriolet", "limousine",
    "hybrid", "4wd", "2wd", "hardtop", "targa", "fastback", "sportback",
]
# 已知的跨品牌同平台关系（可作为"平台共享"判定的补充）
PLATFORM_EQUIV = [
    {"dodge sprinter", "mercedes-benz sprinter", "freightliner sprinter"},
    {"gmc savana", "chevrolet express"},
    {"toyota camry", "lexus es"},
    {"honda accord", "acura tl"},
    {"toyota land cruiser", "lexus lx"},
    {"nissan armada", "infiniti qx56", "infiniti qx80"},
    {"ford explorer", "mercury mountaineer"},
    {"audi a5", "audi s5", "audi rs5"},
    {"bmw 6 series", "bmw m6"},
    {"audi tt", "audi tts", "audi tt rs"},
    {"chevrolet silverado 1500", "chevrolet silverado 2500hd", "gmc sierra 1500"},
    {"audi 100", "audi v8"},
    {"rolls-royce phantom", "rolls-royce ghost"},
]

YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


def split_year(name: str) -> tuple[str, int | None]:
    m = YEAR_RE.search(name)
    if not m:
        return name.strip(), None
    return name[: m.start()].strip(), int(m.group(1))


def lineage(name: str) -> str:
    """剥离年份与车身/配置词，得到车型血脉（小写）。"""
    base, _ = split_year(name)
    s = " " + base.lower() + " "
    for w in sorted(BODY_WORDS, key=len, reverse=True):
        s = s.replace(f" {w} ", " ")
    return re.sub(r"\s+", " ", s).strip()


def brand(name: str) -> str:
    base, _ = split_year(name)
    return base.split()[0].lower() if base else ""


def on_platform(a: str, b: str) -> bool:
    la, lb = lineage(a), lineage(b)
    for grp in PLATFORM_EQUIV:
        if any(g in la for g in grp) and any(g in lb for g in grp):
            return True
    # 同一血脉的跨品牌（如 audi a5 / audi s5 已由 lineage 处理；此处兜底）
    return False


def classify(a: str, b: str) -> str:
    la, lb = lineage(a), lineage(b)
    _, ya = split_year(a)
    _, yb = split_year(b)
    if la == lb:
        return "①同车系·仅年份不同" if ya != yb else "①同车系·同年份"
    if brand(a) == brand(b):
        return "②同品牌·车型相近"
    if on_platform(a, b):
        return "③同平台·跨品牌换标"
    return "④不同车型"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--top", type=int, default=100)
    args = ap.parse_args()

    csv_path = args.csv
    if csv_path is None:
        cands = sorted(RESULTS.glob("test8_混淆对_*.csv"),
                       key=lambda p: p.stat().st_mtime)
        cands = [c for c in cands if "结构" not in c.name]
        if not cands:
            raise SystemExit(f"[FAIL] 在 {RESULTS} 下找不到 test8_混淆对_*.csv")
        csv_path = cands[-1]
    print(f"[输入] {csv_path}")

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))[: args.top]
    print(f"       共 {len(rows)} 对混淆（按错误次数降序）\n")

    cats: Counter = Counter()
    total = 0
    year_gaps: list[int] = []
    detail = []
    for r in rows:
        a, b, n = r["真实类别"], r["被误认成"], int(r["错误次数"])
        c = classify(a, b)
        cats[c] += n
        total += n
        if c == "①同车系·仅年份不同":
            ya, yb = split_year(a)[1], split_year(b)[1]
            if ya and yb:
                year_gaps.append(abs(ya - yb))
        detail.append((n, a, b, c))

    print("=" * 100)
    print("【混淆结构归类】按错误次数统计")
    print("=" * 100)
    for k in sorted(cats):
        v = cats[k]
        bar = "█" * max(1, int(v / max(total, 1) * 40))
        print(f"  {k:<18} {v:>5} 次  {v/total*100:>5.1f}%  {bar}")
    print(f"  {'合计':<18} {total:>5} 次")

    same_family = cats["①同车系·仅年份不同"] + cats["①同车系·同年份"] + cats["②同品牌·车型相近"]
    print(f"\n  → 「同车系/同品牌（视觉近乎相同）」合计 {same_family} 次 = {same_family/total*100:.1f}%")
    print(f"  → 「跨品牌同平台换标」{cats['③同平台·跨品牌换标']} 次 = "
          f"{cats['③同平台·跨品牌换标']/total*100:.1f}%")
    print(f"  → 「不同车型」{cats['④不同车型']} 次 = {cats['④不同车型']/total*100:.1f}%")
    if year_gaps:
        print(f"\n  → 仅年份不同的混淆，平均年份差 = {sum(year_gaps)/len(year_gaps):.1f} 年")

    print("\n" + "=" * 100)
    print("【明细】前 25 对")
    print("=" * 100)
    for n, a, b, c in detail[:25]:
        print(f"  {n:>3} 次  [{c}]")
        print(f"          {a}")
        print(f"       →  {b}")

    out = RESULTS / "test8_混淆结构分析.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"混淆结构分析 · 输入 {csv_path.name} · 分析 {len(rows)} 对\n\n")
        for k in sorted(cats):
            f.write(f"{k}\t{cats[k]}\t{cats[k]/total*100:.1f}%\n")
        f.write(f"\n同车系/同品牌合计\t{same_family}\t{same_family/total*100:.1f}%\n")
        if year_gaps:
            f.write(f"平均年份差\t{sum(year_gaps)/len(year_gaps):.2f}\n")
        f.write("\n明细:\n")
        for n, a, b, c in detail:
            f.write(f"{n}\t{c}\t{a}\t{b}\n")
    print(f"\n[输出] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
