# -*- coding: utf-8 -*-
r"""
工具层：本 agent 的"手足"

三个工具：
  classify_local(image)  —— 本地 Car-1000 模型识别（离线，25.87 MB，不联网）
  analyze_vision(image)  —— 调用云端多模态大模型识别（需联网）
  search_specs(query)    —— 搜索车辆参数（Tavily，需联网）

设计要点：
  · 工具只负责"取数据"，返回**纯文本**给 LLM 阅读
  · 图片不进 LLM 对话上下文，避免每轮重复传输
  · 每个工具都做异常兜底，返回错误文本而不是抛异常（保证 agent 循环不崩）
"""
from __future__ import annotations

import base64
import csv
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import config_loader as cfg

# ---------- 本地模型（懒加载：第一次用到才载入，加快启动）----------
_sess = None
_labels: list[str] | None = None
_taxonomy: dict[str, tuple[str, str]] = {}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 224


def _load_labels() -> list[str]:
    global _labels
    if _labels is not None:
        return _labels
    p = cfg.LABELS_CN if cfg.LABELS_CN.exists() else cfg.LABELS_EN
    if not p.exists():
        _labels = []
        return _labels
    with open(p, encoding="utf-8") as f:
        _labels = [l.strip() for l in f if l.strip()]
    return _labels


def _load_taxonomy() -> dict[str, tuple[str, str]]:
    """class_id -> (中文名, 大类)。用于把类别序号映射回大类。"""
    global _taxonomy
    if _taxonomy:
        return _taxonomy
    if not cfg.TAXONOMY.exists():
        return _taxonomy
    with open(cfg.TAXONOMY, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            _taxonomy[r["name_cn"]] = (r["name_cn"], r["big_type"])
    return _taxonomy


def _load_session():
    global _sess
    if _sess is not None:
        return _sess
    import onnxruntime as ort
    if not cfg.ONNX_PATH.exists():
        raise FileNotFoundError(f"模型不存在: {cfg.ONNX_PATH}")
    opts = ort.SessionOptions()
    opts.log_severity_level = 3          # 静音
    _sess = ort.InferenceSession(str(cfg.ONNX_PATH), sess_options=opts,
                                 providers=["CPUExecutionProvider"])
    return _sess


# ============================================================
# 工具 1：本地经典深度学习模型（离线）
# ============================================================
def classify_local(image_bytes: bytes, topk: int = 5) -> str:
    """
    用本地训练的 Car-1000 模型识别车型。
    完全离线、不联网、零成本；25.87 MB 模型，CPU 推理约 17 ms。
    """
    try:
        sess = _load_session()          # 首次调用含模型加载（约 0.4s），之后走缓存
        labels = _load_labels()
        if not labels:
            return "错误：类别表缺失，无法给出类别名。"

        im = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(
            (INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        x = np.asarray(im, dtype=np.float32) / 255.0
        x = (x - MEAN) / STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1))[None].astype(np.float32)

        t0 = time.time()                # 只计纯推理时间
        out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
        ms = (time.time() - t0) * 1000

        # softmax（数值稳定）
        e = np.exp(out - out.max())
        p = e / e.sum()
        order = p.argsort()[::-1][:topk]

        tax = _load_taxonomy()
        lines = [f"本地模型识别结果（Car-1000，1000 类；离线推理 {ms:.0f} ms，不联网）："]
        for r, i in enumerate(order, 1):
            name = labels[i] if i < len(labels) else f"class_{i}"
            big = tax.get(name, ("", ""))[1]
            lines.append(f"  Top{r}: {name} [{big}]  置信度 {p[i]*100:.2f}%")
        # 大类汇总（模型的强项：大类准确率 98.24%）
        best_big = tax.get(labels[order[0]], ("", ""))[1]
        lines.append(f"  → 判断车辆大类：{best_big}（该模型大类准确率 98.24%）")
        return "\n".join(lines)
    except Exception as e:
        return f"错误：本地模型推理失败 - {type(e).__name__}: {e}"


# ============================================================
# 工具 2：云端多模态大模型（联网）
# ============================================================
def analyze_vision(image_bytes: bytes, question: str = "") -> str:
    """
    把图片交给云端多模态大模型分析。
    ⚠️ 必须给足 max_tokens：该模型带思考链，预算不足会导致正文被截断（返回空）。
    """
    if not cfg.has_llm():
        return "错误：未配置 DeepSeek API 密钥，无法调用云端模型。"
    if not image_bytes:
        return "错误：没有收到图片。"
    try:
        from openai import OpenAI
        cli = OpenAI(api_key=cfg.get("deepseek_key"), base_url=cfg.get("deepseek_base"))

        # 压缩一下，减少传输量（长边不超过 1024）
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if max(im.size) > 1024:
            r = 1024 / max(im.size)
            im = im.resize((int(im.width * r), int(im.height * r)), Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88)
        b64 = base64.b64encode(buf.getvalue()).decode()

        q = question or ("请识别这张图片里的汽车，按以下格式回答：\n"
                         "品牌：\n车系：\n车型：\n"
                         "车身类型（轿车/SUV/MPV/卡车/跑车/微面/轻客）：\n"
                         "把握度（高/中/低）：")
        t0 = time.time()
        resp = cli.chat.completions.create(
            model=cfg.get("deepseek_model"),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": q},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            max_tokens=1500,          # ⚠️ 不能小，否则思考链吃光预算
        )
        ch = resp.choices[0]
        content = (ch.message.content or "").strip()
        if not content:
            return (f"错误：云端模型返回空内容（finish_reason={ch.finish_reason}）。"
                    f"通常因为 max_tokens 不足。")
        return (f"云端多模态大模型识别结果（联网，耗时 {time.time()-t0:.1f} s）：\n"
                f"{content}")
    except Exception as e:
        return f"错误：调用云端视觉模型失败 - {type(e).__name__}: {e}"


# ============================================================
# 工具 3：搜索车辆参数（Tavily，联网）
# ============================================================
def search_specs(query: str) -> str:
    """
    搜索某款车的参数、价格、动力形式等实时信息。
    用途：补足 LLM 知识冻结的缺口 —— 它不知道当前车价和最新配置。
    """
    if not cfg.has_search():
        return "错误：未配置 Tavily API 密钥，无法联网搜索。"
    if not query or not query.strip():
        return "错误：搜索关键词为空。"
    try:
        from tavily import TavilyClient
        tv = TavilyClient(api_key=cfg.get("tavily_key"))
        t0 = time.time()
        r = tv.search(query=f"{query} 参数 配置 价格",
                      search_depth="basic", include_answer=True, max_results=3)
        out = [f"搜索「{query}」结果（耗时 {time.time()-t0:.1f} s）："]
        if r.get("answer"):
            out.append(f"  综合：{r['answer']}")
        for i, item in enumerate(r.get("results", [])[:3], 1):
            title = (item.get("title") or "")[:80]
            content = (item.get("content") or "")[:200].replace("\n", " ")
            out.append(f"  [{i}] {title}")
            out.append(f"      {content}")
        if len(out) == 1:
            return f"搜索「{query}」没有找到有效结果。"
        return "\n".join(out)
    except Exception as e:
        return f"错误：搜索失败 - {type(e).__name__}: {e}"


# ============================================================
# 工具注册表
# ============================================================
TOOLS = {
    "classify_local": classify_local,
    "analyze_vision": analyze_vision,
    "search_specs": search_specs,
}

TOOL_DESC = """
- `classify_local()`: 用**本地离线模型**识别车型（25.87 MB，不联网，1000 类，约 17ms）。
  适合：需要离线/实时/零成本的场景。返回 Top-5 类别和大类判断。
- `analyze_vision()`: 用**云端多模态大模型**识别车型（需联网，秒级）。
  适合：需要更强通用理解、或需要自然语言描述的场合。
- `search_specs(query: str)`: 联网搜索某款车的参数、价格、动力形式。
  适合：需要实时信息（大模型自己不知道当前车价/最新配置）。
"""
