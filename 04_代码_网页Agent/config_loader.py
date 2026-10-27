# -*- coding: utf-8 -*-
r"""
配置：API 密钥与模型路径

⚠️ 密钥优先级：环境变量 > config.ini
   建议把密钥放在 config.ini（已被 .gitignore 排除），不要硬编码进代码。

复制 config.example.ini 为 config.ini 后填写即可。
"""
from __future__ import annotations

import configparser
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "config.ini"

# ---------- 模型相关路径（自动探测多个可能位置）----------
# 之所以要做候选列表：开发目录与分发给别人的目录结构不同，
# 且模型可能放在 deploy/ 或 models/ 下。找不到就报清楚。
HERE = Path(__file__).resolve().parent          # .../agent
_AGENT_PARENT = HERE.parent                     # 开发: new datasets model ; 分发: 工具根目录

# 优先原版 FP32（精度最高）；其次 INT8 动态量化版（体积仅 1/4，随仓库分发）
# 两者输入输出规格完全一致（float32, [1,3,224,224] → [1,1000]），可直接替换。
# 实测差异（193 张测试图）：FP32 94.30% / INT8 91.71%，一致率 96.37%。
_MODEL_NAMES = [
    "cars1000_resnet50_224.onnx",        # FP32，97.41 MB
    "cars1000_resnet50_224_int8.onnx",   # INT8 动态量化，24.53 MB（仓库随附）
]

# 模型可能所在的目录（按优先级）
_MODEL_DIRS = [
    _AGENT_PARENT / "models",            # 分发结构: <根>/models/xxx.onnx
    _AGENT_PARENT / "deploy",            # 开发结构: new datasets model/deploy/
    _AGENT_PARENT / "05_模型文件",        # 本仓库结构: <repo>/05_模型文件/
    HERE / "models",                     # 模型放在 agent/ 里
    HERE,                                # 模型和代码同目录
]

# 先生成「所有目录的 FP32」，再生成「所有目录的 INT8」→ 保证 FP32 优先
_MODEL_CANDIDATES = [d / n for n in _MODEL_NAMES for d in _MODEL_DIRS]

_LABEL_CANDIDATES = []
_TAXO_CANDIDATES = []
for _base in (_AGENT_PARENT / "models", _AGENT_PARENT / "deploy",
              _AGENT_PARENT / "05_模型文件", HERE / "models", HERE):
    _LABEL_CANDIDATES.append(_base / "class_names_cn.txt")
    # 大类映射表：两种文件名都可能出现，字段（name_cn / big_type）一致
    _TAXO_CANDIDATES.append(_base / "class_taxonomy.csv")
    _TAXO_CANDIDATES.append(_base / "class_names_full.csv")
# 开发目录里 taxonomy 在 data/Car-1000/ 下
_TAXO_CANDIDATES.append(_AGENT_PARENT / "data" / "Car-1000" / "class_taxonomy.csv")


def _first_existing(paths: list[Path], default: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    return default          # 都不存在时返回首选路径（便于报错时显示）


ONNX_PATH = _first_existing(_MODEL_CANDIDATES, _MODEL_CANDIDATES[0])
LABELS_CN = _first_existing(_LABEL_CANDIDATES, _LABEL_CANDIDATES[0])
LABELS_EN = LABELS_CN.parent / "class_names_en.txt"
TAXONOMY = _first_existing(_TAXO_CANDIDATES, _TAXO_CANDIDATES[0])

# ---------- 默认值 ----------
DEFAULTS = {
    "deepseek_key": "",
    "deepseek_base": "https://api.deepseek.com",
    "deepseek_model": "deepseek-flash",
    "tavily_key": "",
    "host": "127.0.0.1",
    "port": "8000",
}

# 环境变量名（优先级高于 config.ini）
ENV_MAP = {
    "deepseek_key": "DEEPSEEK_API_KEY",
    "tavily_key": "TAVILY_API_KEY",
    "deepseek_base": "DEEPSEEK_BASE_URL",
    "deepseek_model": "DEEPSEEK_MODEL",
}


def _read_ini() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    cp = configparser.ConfigParser()
    try:
        cp.read(CONFIG_FILE, encoding="utf-8")
    except Exception:
        return {}
    out = {}
    for sec in cp.sections():
        for k, v in cp.items(sec):
            if v and v.strip():
                out[k.strip()] = v.strip()
    return out


_ini = _read_ini()


def get(name: str) -> str:
    """取配置：环境变量 > config.ini > 默认值"""
    env = ENV_MAP.get(name)
    if env and os.environ.get(env):
        return os.environ[env].strip()
    if name in _ini:
        return _ini[name]
    return DEFAULTS.get(name, "")


def has_llm() -> bool:
    return bool(get("deepseek_key"))


def has_search() -> bool:
    return bool(get("tavily_key"))


def summary() -> str:
    """给启动时打印的配置摘要（不泄露密钥全文）"""
    def mask(s):
        return (s[:6] + "..." + s[-4:]) if len(s) > 12 else ("已配置" if s else "未配置")

    return (
        f"  DeepSeek : {mask(get('deepseek_key'))}  ({get('deepseek_model')})\n"
        f"  Tavily   : {mask(get('tavily_key'))}\n"
        f"  本地模型 : {'✅' if ONNX_PATH.exists() else '❌ 缺失'} {ONNX_PATH.name}\n"
        f"  类别表   : {'✅' if LABELS_CN.exists() else '❌ 缺失'} "
    )
