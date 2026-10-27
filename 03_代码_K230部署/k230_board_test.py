# -*- coding: utf-8 -*-
r"""
K230D 板端实测记录工具（脱机运行，把结果写到 TF 卡）

用途：留下**板端实测证据** —— 自动记录
      模型信息、FPS 统计、识别结果，存成 /sdcard/board_test_report.txt

用法：
    1. 本文件改名为 main.py，放到 TF 卡根目录（覆盖原来的）
    2. 按住 KEY2 + 复位，脱机运行（可接充电宝）
    3. 对着车辆图片/实车运行 **60 秒**，程序会自动退出
    4. 把 TF 卡插读卡器，打开 board_test_report.txt 即为实测报告

    ⚠️ 原有 k230_main_v2.py 记得另存为 main_v2_bak.py，测完换回来。
"""

import os
import gc
import time
import math

import nncase_runtime as nn
import ulab.numpy as np
import image

from media.media import *
from media.sensor import *
from libs.PipeLine import PipeLine, ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d

try:
    ALIGN_UP
except NameError:
    def ALIGN_UP(x, n):
        return (x + n - 1) // n * n

# ============================ 配置 ============================
KMODEL_PATH = "/sdcard/cars1000_resnet50_224.kmodel"
LABEL_CN = "/sdcard/class_names_cn.txt"
LABEL_EN = "/sdcard/class_names_en.txt"
REPORT_PATH = "/sdcard/board_test_report.txt"

MODEL_INPUT_SIZE = [224, 224]
RGB888P_SIZE = [640, 480]
DISPLAY_SIZE = [640, 480]
TOP_K = 3
TEST_SECONDS = 60            # 实测时长（秒）
BOX_MARGIN_X, BOX_MARGIN_Y = 0.12, 0.20
# =============================================================


def file_exists(p):
    try:
        os.stat(p)
        return True
    except Exception:
        return False


def load_labels():
    for p in (LABEL_CN, LABEL_EN):
        if file_exists(p):
            out = []
            with open(p, "r") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        out.append(s)
            print("[标签] %s  %d 类" % (p, len(out)))
            return out
    print("[ERROR] 找不到类别表")
    return None


def _now_str():
    """时间字符串。⚠️ MicroPython 不一定支持 time.strftime，做兜底。"""
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    try:
        import utime
        t = utime.localtime()
        return "%04d-%02d-%02d %02d:%02d:%02d" % (t[0], t[1], t[2], t[3], t[4], t[5])
    except Exception:
        return "(时间不可用)"


def softmax(vals):
    if not vals:
        return vals
    m = max(vals)
    ex = []
    for v in vals:
        try:
            ex.append(math.exp(v - m))
        except Exception:
            ex.append(1.0)
    s = 0.0
    for e in ex:
        s += e
    if s <= 0:
        s = 1e-30
    return [e / s for e in ex]


def topk(vals, k):
    n = len(vals)
    k = k if k < n else n
    order = sorted(range(n), key=lambda i: vals[i], reverse=True)[:k]
    return [(int(i), float(vals[i])) for i in order]


class VehicleApp(AIBase):
    def __init__(self, kmodel_path, model_input_size, labels,
                 rgb888p_size=None, display_size=None, debug_mode=0):
        rgb888p_size = rgb888p_size or RGB888P_SIZE
        display_size = display_size or DISPLAY_SIZE
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.labels = labels
        self.model_input_size = model_input_size
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        _u8 = np.uint8 if hasattr(np, "uint8") else "B"
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,
                                 nn.ai2d_format.NCHW_FMT, _u8, _u8)

    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            size = input_image_size if input_image_size else self.rgb888p_size
            self.ai2d.resize(nn.interp_method.tf_bilinear,
                             nn.interp_mode.half_pixel)
            self.ai2d.build([1, 3, size[1], size[0]],
                            [1, 3, self.model_input_size[1],
                             self.model_input_size[0]])

    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            raw = results[0]
            if hasattr(raw, "tolist"):
                raw = raw.tolist()
            raw = list(raw)
            while raw and isinstance(raw[0], (list, tuple)):
                raw = list(raw[0])
            return softmax([float(x) for x in raw])

    def draw_result(self, pl, probs, fps):
        osd = pl.osd_img
        osd.clear()
        dw, dh = self.display_size[0], self.display_size[1]
        x1 = int(dw * BOX_MARGIN_X); y1 = int(dh * BOX_MARGIN_Y)
        x2 = int(dw * (1 - BOX_MARGIN_X)); y2 = int(dh * (1 - BOX_MARGIN_Y))
        if probs is None:
            osd.draw_rectangle(x1, y1, x2 - x1, y2 - y1,
                               color=(255, 255, 255, 0), thickness=3)
            return
        ci, cv = topk(probs, 1)[0]
        name = self.labels[ci] if ci < len(self.labels) else "?"
        color = (255, 80, 220, 80) if cv >= 0.5 else (255, 255, 165, 0)
        osd.draw_rectangle(x1, y1, x2 - x1, y2 - y1, color=color, thickness=4)
        ty = y1 - 46
        if ty < 4:
            ty = y1 + 6
        osd.draw_string_advanced(x1 + 6, ty, 32, name, color=color)
        osd.draw_string_advanced(x1 + 6, ty + 36, 26,
                                 "%.1f%%" % (cv * 100.0), color=color)
        osd.draw_string_advanced(6, 6, 22, "FPS %.1f" % fps,
                                 color=(255, 255, 255, 0))


def main():
    print("=== K230D 板端实测记录 ===")
    labels = load_labels()
    if labels is None or not file_exists(KMODEL_PATH):
        print("[ERROR] 缺少模型或类别表")
        return

    display_mode = "lcd"
    sensor = Sensor(width=1280, height=960)
    pl = PipeLine(rgb888p_size=RGB888P_SIZE, display_size=DISPLAY_SIZE,
                  display_mode=display_mode)
    pl.create(sensor=sensor)

    app = None
    log = []
    t_start = time.time()
    try:
        app = VehicleApp(KMODEL_PATH, MODEL_INPUT_SIZE, labels,
                         rgb888p_size=RGB888P_SIZE,
                         display_size=DISPLAY_SIZE, debug_mode=0)
        app.config_preprocess()

        # 预热（丢弃前几帧，避免首帧异常值影响统计）
        for _ in range(5):
            app.run(pl.get_frame())

        t_last = time.time()
        frames = 0
        fps_list = []
        cur_fps = 0.0
        top1_count = {}
        conf_list = []

        while True:
            os.exitpoint()
            img = pl.get_frame()
            probs = app.run(img)
            app.draw_result(pl, probs, cur_fps)
            pl.show_image()

            if probs is not None:
                ci, cv = topk(probs, 1)[0]
                nm = labels[ci] if ci < len(labels) else "?"
                top1_count[nm] = top1_count.get(nm, 0) + 1
                conf_list.append(cv)

            frames += 1
            now = time.time()
            if now - t_last >= 1.0:
                cur_fps = frames / (now - t_last)
                fps_list.append(cur_fps)
                frames, t_last = 0, now
                if probs is not None:
                    ci, cv = topk(probs, 1)[0]
                    nm = labels[ci] if ci < len(labels) else "?"
                    line = "FPS %.1f | %s %.1f%%" % (cur_fps, nm, cv * 100.0)
                    print(line)
                    log.append(line)
                gc.collect()

            if now - t_start >= TEST_SECONDS:
                print("[实测时长到，退出]")
                break

    except KeyboardInterrupt as e:
        print("用户停止:", e)
    except BaseException as e:
        print("异常:", e)
    finally:
        # ---------- 写报告 ----------
        try:
            lines = []
            lines.append("=" * 62)
            lines.append("K230D 板端实测报告")
            lines.append("=" * 62)
            lines.append("生成时间: %s" % _now_str())
            lines.append("")
            lines.append("【实验配置】")
            lines.append("  模型      : %s" % KMODEL_PATH)
            lines.append("  模型输入  : %s" % str(MODEL_INPUT_SIZE))
            lines.append("  AI 帧尺寸 : %s" % str(RGB888P_SIZE))
            lines.append("  显示尺寸  : %s" % str(DISPLAY_SIZE))
            lines.append("  类别数    : %d" % (len(labels) if labels else 0))
            lines.append("  实测时长  : %d 秒" % TEST_SECONDS)
            try:
                st = os.stat(KMODEL_PATH)
                lines.append("  kmodel 大小: %d 字节 (%.2f MB)"
                             % (st[6], st[6] / 1024.0 / 1024.0))
            except Exception:
                pass
            lines.append("")
            lines.append("【性能指标】")
            if fps_list:
                lines.append("  FPS 平均  : %.2f" % (sum(fps_list) / len(fps_list)))
                lines.append("  FPS 最大  : %.2f" % max(fps_list))
                lines.append("  FPS 最小  : %.2f" % min(fps_list))
                lines.append("  有效秒数  : %d" % len(fps_list))
            else:
                lines.append("  （无 FPS 数据）")
            lines.append("")
            lines.append("【识别统计】")
            if conf_list:
                lines.append("  平均置信度: %.2f%%"
                             % (sum(conf_list) / len(conf_list) * 100.0))
                lines.append("  最高置信度: %.2f%%" % (max(conf_list) * 100.0))
                lines.append("  最低置信度: %.2f%%" % (min(conf_list) * 100.0))
            lines.append("")
            lines.append("  Top-1 类别分布（出现次数）:")
            total = sum(top1_count.values())
            for nm, c in sorted(top1_count.items(), key=lambda x: -x[1])[:15]:
                lines.append("    %-28s %4d 次  %.1f%%"
                             % (nm, c, c * 100.0 / total if total else 0))
            lines.append("")
            lines.append("【逐秒记录】")
            for e in log:
                lines.append("  " + e)
            lines.append("")
            lines.append("=" * 62)

            with open(REPORT_PATH, "w") as f:
                f.write("\n".join(lines))
            print("[报告已保存] %s" % REPORT_PATH)
        except Exception as e:
            print("[写报告失败]", e)

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
