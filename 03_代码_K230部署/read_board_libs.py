# 在 CanMV IDE 串行终端里粘贴运行（REPL）
# 目的：读出板子上真实的 PipeLine / AIBase 源码，确认 get_frame() 给的是什么格式

import os

for f in ["/sdcard/libs/PipeLine.py", "/sdcard/libs/AIBase.py"]:
    print("=" * 70)
    print("FILE:", f)
    print("=" * 70)
    try:
        with open(f, "r") as fp:
            src = fp.read()
        print(src)
    except Exception as e:
        print("读取失败:", e)
