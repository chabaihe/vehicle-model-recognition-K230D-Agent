# -*- coding: utf-8 -*-
"""验证 nncase 的 k230 target 是否注册成功（KPU 插件是否生效）"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import nncase

print("nncase 导入成功")
print("  __file__:", nncase.__file__)

opts = nncase.CompileOptions()
try:
    opts.target = "k230"
    print("  设置 target=k230 成功")
except Exception as e:
    print("  设置 target 失败:", e)
    sys.exit(1)

try:
    c = nncase.Compiler(opts)
    print("  Compiler 创建成功 → k230 target 已注册，KPU 插件生效！")
except Exception as e:
    print("  Compiler 创建失败:", e)
    sys.exit(1)
