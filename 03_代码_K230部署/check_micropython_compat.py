# -*- coding: utf-8 -*-
"""
板端脚本 MicroPython 兼容性静态检查

背景：PC 上的 CPython 功能齐全，MicroPython 是精简子集。
      第一次上板就踩了 os.path.exists 不存在 的坑。
      本脚本用 AST + 规则扫描，提前找出板端会报错的写法。

用法：
    python check_micropython_compat.py <板端脚本路径>
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# MicroPython 上没有 stdlib 模块（导入即报错）
# 注意：math / cmath / random / re（部分固件）/ json（部分）情况不同，
#      这里只列**确定没有**的；math 在 MicroPython 是支持的，不要列进来。
BANNED_MODULES = {
    "pathlib", "typing", "dataclasses", "enum", "collections",
    "functools", "itertools", "csv", "argparse", "logging",
    "subprocess", "shutil", "tempfile", "glob", "threading",
    "queue", "copy", "pickle", "numpy", "scipy", "pandas",
}

# MicroPython 上不存在的 os / 常见模块属性
BANNED_ATTRS = {
    "os": {"path", "system", "popen", "environ", "getcwd", "chdir", "makedirs", "walk"},
    "sys": {"argv", "exit"},
    "time": {"time_ns", "perf_counter", "monotonic"},
}

# CPython 才有的内置函数
BANNED_BUILTINS = {"input", "print_", "help", "breakpoint", "vars", "dir"}

# ulab（板端的 numpy）**不导出**的 dtype/常量
# 依据 ulab 官方文档：dtype 常量只有 bool/int8/int16/uint8/uint16/float
ULAB_MISSING_ATTRS = {
    "float32", "float64", "double", "single", "half",
    "int32", "int64", "uint32", "uint64", "long", "intp", "uintp",
    "complex", "complex64", "complex128", "longdouble",
    "nan", "inf", "e", "pi",
}

# ulab 存在但**限制较多**的函数（用错参数会运行时报错）
ULAB_CAUTION = {
    "argsort": "ulab 的 argsort 不支持二维数组，会报 'not implemented for flattened arrays'",
    "sort": "ulab 的 sort 不支持二维数组/axis 参数",
    "median": "ulab 不支持 median",
    "percentile": "ulab 不支持 percentile",
    "unique": "ulab 不支持 unique",
    "mean": "ulab 的 mean 不支持 axis 参数",
    "std": "ulab 的 std 不支持 axis 参数",
    "linspace": "ulab 支持，但注意 dtype 参数",
}


def _is_np_module(node: ast.AST) -> bool:
    """判断属性访问的对象是不是 np / numpy"""
    return isinstance(node, ast.Name) and node.id in ("np", "numpy", "ulab")


def check(path: Path) -> int:
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"  [语法错误] 行 {e.lineno}: {e.msg}")
        return 1

    issues: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        # --- import 检查 ---
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root in BANNED_MODULES:
                    issues.append((node.lineno, f"导入了 MicroPython 不支持的模块: {a.name}"))
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in BANNED_MODULES:
                issues.append((node.lineno, f"导入了 MicroPython 不支持的模块: {node.module}"))

        # --- 属性访问检查 ---
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                mod = node.value.id
                if mod in BANNED_ATTRS and node.attr in BANNED_ATTRS[mod]:
                    issues.append((node.lineno, f"MicroPython 不支持: {mod}.{node.attr}"))
                # --- ulab 专有检查 ---
                if _is_np_module(node.value):
                    if node.attr in ULAB_MISSING_ATTRS:
                        issues.append(
                            (node.lineno, f"ulab 不导出 np.{node.attr}（会报 no attribute）"))
                    elif node.attr in ULAB_CAUTION:
                        issues.append(
                            (node.lineno, f"注意 np.{node.attr}：{ULAB_CAUTION[node.attr]}"))

        # --- 函数调用检查 ---
        elif isinstance(node, ast.Call):
            fn = node.func
            # open(..., encoding=) —— MicroPython 的 open 不支持 encoding 关键字
            if isinstance(fn, ast.Name) and fn.id == "open":
                for kw in node.keywords:
                    if kw.arg == "encoding":
                        issues.append(
                            (node.lineno, "MicroPython 的 open() 不支持 encoding= 参数（需删掉）"))
            if isinstance(fn, ast.Name) and fn.id in BANNED_BUILTINS:
                issues.append((node.lineno, f"CPython 专有内置函数: {fn.id}()"))

    if not issues:
        print("  ✅ 未发现 MicroPython 兼容性问题")
        return 0

    print(f"  ⚠️ 发现 {len(issues)} 处潜在问题：")
    for ln, msg in sorted(set(issues)):
        print(f"     行 {ln:>4}: {msg}")
    return 1


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).resolve().parent.parent / "deploy" / "k230_main_v2.py"
    print(f"[检查] {target}")
    raise SystemExit(check(target))
