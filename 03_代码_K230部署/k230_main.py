# -*- coding: utf-8 -*-
r"""
K230D 板端实时车型识别（Car-1000，1000 类）—— v3

基于正点原子 CanMV AI 框架：
    PipeLine（摄像头 + 显示）+ AIBase（模型加载/推理）+ Ai2d（预处理）

v3 相对 v2 的修改（实测反馈驱动）：
  1. **加 softmax** —— v2 直接输出 logits，屏幕上显示成 -520.5% 这种负数
  2. **rgb888p_size 改为 [640,480]** —— v2 用 [224,224] 导致视野过小
  3. **UI 改为固定车辆框 + 只显示 Top-1** —— 按需求（路线 A）

⚠️ 板端是 MicroPython + ulab，与 PC 上的 CPython + numpy 差异很大。
   改代码前请先跑 code/check_micropython_compat.py 做静态检查。

部署（TF 卡根目录）：
    /sdcard/cars1000_resnet50_224.kmodel    25.87 MB
    /sdcard/class_names_cn.txt              1000 行
    /sdcard/main.py                         本文件改名

启动：按住 KEY2 + 复位（从 TF 卡启动）
"""

import os
import gc
import sys
import time

# math 在 MicroPython 里是支持的，但再做一层兜底（个别精简固件可能裁掉）
try:
    import math
except ImportError:
    math = None

import nncase_runtime as nn
import ulab.numpy as np
import image

from media.media import *
from media.sensor import *
from libs.PipeLine import PipeLine, ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d

# ALIGN_UP 在官方例程里是全局可用的（由 PipeLine/AI2D 注入）。
# 但为了稳妥，这里做一次兜底定义：只有不存在时才定义。
try:
    ALIGN_UP
except NameError:
    def ALIGN_UP(x, n):
        return (x + n - 1) // n * n

# ============================ 配置区 ============================
# 模型路径：候选列表，自动找存在的那个（CanMV IDE 上传时落点不定，省得改代码）
KMODEL_CANDIDATES = (
    "/sdcard/cars1000_resnet50_224.kmodel",
    "/sdcard/kmodel/cars1000_resnet50_224.kmodel",
    "/sdcard/examples/kmodel/cars1000_resnet50_224.kmodel",
    "/sdcard/Car-1000/cars1000_resnet50_224.kmodel",
)
LABEL_CANDIDATES = (
    "/sdcard/class_names_cn.txt",
    "/sdcard/kmodel/class_names_cn.txt",
    "/sdcard/examples/kmodel/class_names_cn.txt",
    "/sdcard/class_names_en.txt",
    "/sdcard/kmodel/class_names_en.txt",
)

MODEL_INPUT_SIZE = [224, 224]     # 模型输入，必须与训练/编译一致
# ⚠️ v3 改动：v2 用的是 [224,224]（视野过小），改成 640×480（与官方例程同量级）
RGB888P_SIZE     = [640, 480]
DISPLAY_SIZE     = [640, 480]     # 板载 LCD
CONF_THRESHOLD   = 0.50           # 低于此值提示"不确定"
DEBUG_MODE       = 0

# ---- 车辆框（路线 A：固定位置的框）----
# 比例值(0~1)，相对 AI 输入画面；0.12 表示左右各留 12% 边距
BOX_MARGIN_X     = 0.12
BOX_MARGIN_Y     = 0.20
# ===============================================================


def file_exists(path):
    """
    判断文件是否存在。

    ⚠️ MicroPython 的 os 模块**没有 os.path**！可用的只有
       os.stat / os.ilistdir / os.remove / os.rename 等。
    """
    try:
        os.stat(path)
        return True
    except OSError:
        pass
    except Exception:
        pass
    try:
        d = path.rsplit("/", 1)
        if len(d) == 2:
            names = [x if isinstance(x, str) else x[0] for x in os.ilistdir(d[0] or "/")]
            return d[1] in names
    except Exception:
        pass
    return False


def find_first(candidates, what):
    for p in candidates:
        if file_exists(p):
            print("[找到%s] %s" % (what, p))
            return p
    print("[ERROR] 找不到%s，已查找以下位置：" % what)
    for p in candidates:
        print("        %s" % p)
    return None


def load_labels():
    path = find_first(LABEL_CANDIDATES, "类别表")
    if path is None:
        return None
    labels = []
    # ⚠️ MicroPython 的 open() 不支持 encoding 参数
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                labels.append(s)
    print("[标签] 共 %d 类" % len(labels))
    return labels


def softmax(vals):
    """数值稳定的 softmax（纯 Python，避开 ulab 限制）。"""
    if not vals:
        return vals
    m = max(vals)
    ex = []
    for v in vals:
        if math is not None:
            ex.append(math.exp(v - m))
        else:
            ex.append(np.exp(v - m))       # 兜底：ulab 的 exp
    s = 0.0
    for e in ex:
        s += e
    if s <= 0:
        s = 1e-30
    return [e / s for e in ex]


class VehicleClassifyApp(AIBase):
    """车型分类：继承 AIBase，基类负责模型加载与推理调度。"""

    def __init__(self, kmodel_path, model_input_size, labels,
                 rgb888p_size=None, display_size=None, debug_mode=0):
        rgb888p_size = rgb888p_size or RGB888P_SIZE
        display_size = display_size or DISPLAY_SIZE
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)

        self.kmodel_path = kmodel_path
        self.labels = labels
        self.model_input_size = model_input_size
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode

        # Ai2d 预处理；注意板端 np 是 ulab，某些 dtype 常量可能不存在
        _u8 = np.uint8 if hasattr(np, "uint8") else "B"
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,
                                 nn.ai2d_format.NCHW_FMT,
                                 _u8, _u8)

    def config_preprocess(self, input_image_size=None):
        """预处理：整幅画面缩放到模型输入尺寸（分类任务只需 resize）。"""
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            size = input_image_size if input_image_size else self.rgb888p_size
            self.ai2d.resize(nn.interp_method.tf_bilinear,
                             nn.interp_mode.half_pixel)
            self.ai2d.build(
                [1, 3, size[1], size[0]],
                [1, 3, self.model_input_size[1], self.model_input_size[0]],
            )

    def postprocess(self, results):
        """
        ⚠️ v3 修复：模型输出是**原始 logits**（范围 ±几百），必须做 softmax
           转成 0~1 概率。v2 漏了这步，屏幕上显示成 -520.5% 这种负数。
        """
        with ScopedTiming("postprocess", self.debug_mode > 0):
            raw = results[0]
            if hasattr(raw, "tolist"):
                raw = raw.tolist()
            raw = list(raw)
            while raw and isinstance(raw[0], (list, tuple)):
                raw = list(raw[0])
            probs = softmax([float(x) for x in raw])
            return probs

    def top1(self, probs):
        """只取置信度最高的一个（路线 A 的需求）。"""
        if not probs:
            return (0, 0.0)
        best_i, best_v = 0, probs[0]
        for i in range(1, len(probs)):
            if probs[i] > best_v:
                best_i, best_v = i, probs[i]
        return (best_i, best_v)

    def draw_result(self, pl, probs):
        """画固定车辆框 + 框上方显示 Top-1（类别名 + 置信度）。"""
        osd = pl.osd_img
        osd.clear()

        dw, dh = self.display_size[0], self.display_size[1]
        # 车辆框（相对 AI 输入画面的比例 → 换算到显示坐标）
        x1 = int(dw * BOX_MARGIN_X)
        y1 = int(dh * BOX_MARGIN_Y)
        x2 = int(dw * (1.0 - BOX_MARGIN_X))
        y2 = int(dh * (1.0 - BOX_MARGIN_Y))

        if probs is None:
            osd.draw_rectangle(x1, y1, x2 - x1, y2 - y1,
                               color=(255, 255, 255, 0), thickness=3)
            osd.draw_string_advanced(x1 + 8, max(0, y1 - 40), 26, "识别中...",
                                     color=(255, 255, 255, 0))
            return

        ci, cv = self.top1(probs)
        name = self.labels[ci] if ci < len(self.labels) else ("class_%d" % ci)

        # 置信度低 → 橙色提醒；正常 → 绿色
        if cv >= CONF_THRESHOLD:
            color = (255, 80, 220, 80)      # 绿
        else:
            color = (255, 255, 165, 0)      # 橙（不确定）

        osd.draw_rectangle(x1, y1, x2 - x1, y2 - y1, color=color, thickness=4)

        # 标签画在框上方：类别名 + 置信度
        ty = y1 - 46
        if ty < 4:
            ty = y1 + 6
        osd.draw_string_advanced(x1 + 6, ty, 32, name,
                                 color=color)
        osd.draw_string_advanced(x1 + 6, ty + 36, 26, "%.1f%%" % (cv * 100.0),
                                 color=color)


def main():
    print("=== K230D 车型识别 v3（Car-1000, 1000 类）===")

    labels = load_labels()
    if labels is None:
        print("[提示] 用 CanMV IDE「保存文件到 CanMV 设备」上传类别表。")
        return

    kmodel_path = find_first(KMODEL_CANDIDATES, "模型文件")
    if kmodel_path is None:
        print("[提示] 用 CanMV IDE「保存文件到 CanMV 设备」上传 kmodel（25.9 MB）。")
        return

    display_mode = "lcd"
    display_size = DISPLAY_SIZE

    sensor = Sensor(width=1280, height=960)
    pl = PipeLine(rgb888p_size=RGB888P_SIZE,
                  display_size=display_size,
                  display_mode=display_mode)
    pl.create(sensor=sensor)

    app = None
    try:
        app = VehicleClassifyApp(kmodel_path, MODEL_INPUT_SIZE, labels,
                                 rgb888p_size=RGB888P_SIZE,
                                 display_size=display_size,
                                 debug_mode=DEBUG_MODE)
        app.config_preprocess()

        t_last = time.time()
        frames = 0
        while True:
            os.exitpoint()

            img = pl.get_frame()
            probs = app.run(img)

            app.draw_result(pl, probs)
            pl.show_image()

            frames += 1
            now = time.time()
            if now - t_last >= 1.0:
                fps = frames / (now - t_last)
                frames, t_last = 0, now
                if probs is not None:
                    ci, cv = app.top1(probs)
                    nm = labels[ci] if ci < len(labels) else "?"
                    print("FPS %.1f | %s  %.1f%%" % (fps, nm, cv * 100.0))
                gc.collect()

    except KeyboardInterrupt as e:
        print("用户停止:", e)
    except BaseException as e:
        print("异常:", e)
    finally:
        # 逐个释放，避免初始化失败时二次报错掩盖真实原因
        try:
            if app is not None:
                app.deinit()
        except Exception:
            pass
        try:
            pl.destroy()
        except Exception:
            pass
        try:
            os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
            time.sleep_ms(100)
        except Exception:
            pass
        try:
            MediaManager.deinit()
        except Exception:
            pass


if __name__ == "__main__":
    os.exitpoint(os.EXITPOINT_ENABLE)
    main()
