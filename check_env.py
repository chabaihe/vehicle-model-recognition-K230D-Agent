# -*- coding: utf-8 -*-
r"""
环境自检脚本 —— 检查本项目所需依赖是否齐全

用法：
    python check_env.py            # 检查主环境（训练 / 推理 / Agent）
    python check_env.py --k230     # 检查 K230 模型转换环境（nncase）

退出码：
    0 = 全部就绪
    1 = 有缺失项（会列出缺什么、怎么装）
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# (import 名, PyPI 包名, 是否必需, 用途说明)
MAIN_DEPS = [
    ("torch",       "torch",          True,  "深度学习框架（训练/推理）"),
    ("torchvision", "torchvision",    True,  "ResNet 等预训练模型"),
    ("numpy",       "numpy",          True,  "数值计算"),
    ("PIL",         "pillow",         True,  "读写图片"),
    ("matplotlib",  "matplotlib",     True,  "预测结果可视化"),
    ("scipy",       "scipy",          True,  "prepare_cars.py 用"),
    ("onnx",        "onnx",           True,  "导出 ONNX"),
    ("onnxruntime", "onnxruntime",    True,  "ONNX 推理（电脑端识别）"),
    ("openai",      "openai",         True,  "网页 Agent 云端识别"),
    ("tavily",      "tavily-python",  True,  "网页 Agent 联网检索"),
    ("onnxsim",     "onnxsim",        False, "ONNX 简化（可选）"),
]

K230_DEPS = [
    ("nncase",      "nncase",         True,  "模型编译器本体"),
    ("numpy",       "numpy",          True,  "数值计算"),
    ("PIL",         "pillow",         True,  "读写校准图片"),
    ("onnx",        "onnx",           True,  "读取 ONNX 模型"),
]
# 注意：nncase-kpu 不能用 import 检查！
# 它不是普通的 Python 模块，而是把 KPU 插件 DLL 装到 nncase/modules/kpu/ 下。
# 正确验证方式 = 真的建一个 target=k230 的 Compiler（见下面 _check_k230_target）。
KPU_PKG = "nncase-kpu"


def _check_k230_target() -> tuple[bool, str]:
    r"""
    验证 KPU 插件是否真正生效。

    只看 pip 列表不够：装上了但插件路径不对时，nncase 仍然会报
        KeyNotFoundException: 'k230' was not present in the dictionary
    唯一可靠的判据是「能不能成功创建 target=k230 的 Compiler」。
    """
    try:
        import nncase
    except Exception as e:
        return False, f"nncase 导入失败: {e}"

    ver = _version("nncase")
    try:
        opts = nncase.CompileOptions()
        opts.target = "k230"
        nncase.Compiler(opts)
        return True, f"nncase {ver} → k230 Compiler 创建成功（KPU 插件已生效）"
    except Exception as e:
        return False, f"nncase {ver} → 无法创建 k230 Compiler: {e}"


def _version(pkg: str) -> str:
    try:
        return md.version(pkg)
    except Exception:
        return "-"


def check(deps, title: str) -> int:
    print("=" * 66)
    print(f"  {title}")
    print("=" * 66)
    print(f"  Python {sys.version.split()[0]}  ({sys.executable})")
    print()

    missing_required = []
    missing_optional = []
    ok_count = 0

    for mod, pkg, required, desc in deps:
        try:
            importlib.import_module(mod)
            ver = _version(pkg)
            print(f"  [OK]    {mod:<13} {ver:<12} {desc}")
            ok_count += 1
        except Exception:
            if required:
                print(f"  [MISS]  {mod:<13} {'-':<12} {desc}")
                missing_required.append(pkg)
            else:
                print(f"  [opt]   {mod:<13} {'-':<12} {desc}（可选，不影响运行）")
                missing_optional.append(pkg)

    total = len(deps)
    print()
    print(f"  合计 {total} 项：就绪 {ok_count}，缺失 {len(missing_required)}（必需）"
          f"，{len(missing_optional)}（可选）")

    # ---------- K230 专项：KPU 插件功能测试 ----------
    if any(m == "nncase" for m, *_ in deps):
        print()
        kpu_ver = _version(KPU_PKG)
        print(f"  KPU 插件包 {KPU_PKG}：{'已安装 ' + kpu_ver if kpu_ver != '-' else '未安装'}")
        if kpu_ver == "-":
            missing_required.append(KPU_PKG)
        ok, msg = _check_k230_target()
        if ok:
            print(f"  [OK]    {msg}")
        else:
            print(f"  [MISS]  {msg}")
            if KPU_PKG not in missing_required:
                missing_required.append(KPU_PKG)

    # 额外检查：CUDA 是否可用
    if any(m == "torch" for m, *_ in deps):
        try:
            import torch
            print()
            if torch.cuda.is_available():
                print(f"  显卡：{torch.cuda.get_device_name(0)}")
                print(f"        CUDA {torch.version.cuda}，可用显存 "
                      f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                if torch.cuda.get_device_properties(0).total_memory < 4 * 1024**3:
                    print("        ⚠️ 显存小于 4 GB，训练可能 OOM，建议调小 batch size")
            else:
                print("  显卡：❌ torch.cuda.is_available() = False")
                print("        说明装的是 CPU 版 PyTorch，训练会慢几十倍。")
                print("        重装命令：")
                print("        pip install torch torchvision --index-url "
                      "https://download.pytorch.org/whl/cu126")
        except Exception:
            pass

    print()
    if missing_required:
        print("-" * 66)
        print("  缺少必需依赖，请执行：")
        print()
        if any(p in ("nncase", "nncase-kpu") for p in missing_required):
            print("      pip install -r requirements-k230.txt")
        else:
            print("      pip install -r requirements.txt")
        print()
        print("  或者直接双击仓库根目录的 install.bat（主环境）")
        print("                      install-k230.bat（转换环境）")
        print("-" * 66)
        return 1

    print("  ✅ 环境就绪，可以开始运行脚本了。")
    return 0


def main() -> int:
    if "--k230" in sys.argv:
        return check(K230_DEPS, "K230 模型转换环境（nncase）")
    return check(MAIN_DEPS, "主环境（训练 / 推理 / 网页 Agent）")


if __name__ == "__main__":
    sys.exit(main())
