# -*- coding: utf-8 -*-
r"""
ReAct 车辆识别 Agent —— 核心循环

架构采用经典 ReAct 模式：

    用户上传车辆图片
        ↓
    LLM 思考（Thought）→ 选择工具（Action）
        ↓
    执行工具（本地模型 / 云端视觉 / 联网搜索）
        ↓
    拿到结果（Observation）→ 塞回对话历史
        ↓
    循环，直到 Action: Finish[最终答案]

关键设计：
  · 图片不进入对话上下文 —— 由工具持有，工具返回**纯文本**给 LLM
    好处：多轮循环时不重复传输图片，省 token 又避免格式问题
  · 每轮产出结构化事件，供网页实时展示"思考过程"
"""
from __future__ import annotations

import re
import time
from typing import Callable

import config_loader as cfg
import tools

SYSTEM_PROMPT = """你是一个车辆识别助手。用户会给你一张汽车照片，你需要识别它并给出有用信息。

# 可用工具
{tool_desc}

# 输出格式要求
你的每次回复必须严格遵循以下格式，包含一对 Thought 和 Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action 的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示
- 每次只输出一对 Thought-Action
- Action 必须在同一行，不要换行
- **优先先用两个识别工具各跑一次，对比云端与本地结果**，这能体现两条技术路线的差异
- 当收集到足够信息时，必须用 Action: Finish[最终答案] 结束
- 最终答案要用中文，简洁清楚，包含：识别结果、两条路线的对比、以及必要的车辆信息

请按上述格式开始输出。
""".format(tool_desc=tools.TOOL_DESC)


def _call_llm(messages: list[dict], max_tokens: int = 1200) -> str:
    from openai import OpenAI
    cli = OpenAI(api_key=cfg.get("deepseek_key"), base_url=cfg.get("deepseek_base"))
    resp = cli.chat.completions.create(
        model=cfg.get("deepseek_model"),
        messages=messages,
        max_tokens=max_tokens,      # ⚠️ 该模型带思考链，预算不能太小
    )
    return (resp.choices[0].message.content or "").strip()


def _parse_action(text: str) -> tuple[str, str]:
    """从 LLM 输出里解析 Action，返回 (类型, 内容)"""
    m = re.search(r"Action:\s*(.*)", text, re.DOTALL)
    if not m:
        # 兜底：模型有时写成 "下一步：xxx()" 或直接给函数调用
        m2 = re.search(r"(?:下一步|行动|调用)[:：]\s*(\w+\s*\([^)]*\))", text)
        if m2:
            return ("tool", m2.group(1).strip())
        # 再兜底：整段里出现已知工具名
        for name in ("classify_local", "analyze_vision", "search_specs"):
            if re.search(rf"\b{name}\s*\(", text):
                mm = re.search(rf"({name}\s*\([^)]*\))", text)
                return ("tool", mm.group(1).strip())
        return ("none", "")
    action = m.group(1).strip()
    if action.startswith("Finish"):
        mm = re.match(r"Finish\[(.*)\]", action, re.DOTALL)
        return ("finish", mm.group(1).strip() if mm else action)
    return ("tool", action)


def _truncate_extra(text: str) -> str:
    """模型可能一次输出多对 Thought-Action，截断只保留第一对"""
    m = re.search(r"(Thought:.*?Action:.*?)"
                  r"(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)",
                  text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def run_agent(image_bytes: bytes,
              user_note: str = "",
              on_event: Callable[[dict], None] | None = None,
              max_rounds: int = 6) -> dict:
    """
    跑一轮完整的 agent。

    image_bytes : 用户上传的图片字节
    user_note   : 用户附加的文字说明（可选）
    on_event    : 每步回调，用于网页实时推送
    返回        : {"answer": str, "steps": [...], "elapsed": float}
    """
    def emit(ev: dict):
        ev["t"] = round(time.time() - t0, 2)
        steps.append(ev)
        if on_event:
            try:
                on_event(ev)
            except Exception:
                pass

    t0 = time.time()
    steps: list[dict] = []

    if not cfg.has_llm():
        return {"answer": "⚠️ 未配置 DeepSeek API 密钥，无法运行 Agent。\n"
                          "请复制 config.example.ini 为 config.ini 并填写密钥。",
                "steps": [], "elapsed": 0.0}

    emit({"type": "info", "msg": "收到图片，开始分析"})

    # 图片由工具持有，不进对话上下文
    ctx = {
        "image": image_bytes,
        "note": user_note,
    }

    task = ("用户上传了一张汽车照片，请识别它并给出有用信息。"
            + (f"\n用户的补充说明：{user_note}" if user_note else ""))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    final_answer = ""
    for rnd in range(1, max_rounds + 1):
        emit({"type": "round", "n": rnd})
        try:
            out = _call_llm(messages)
        except Exception as e:
            emit({"type": "error", "msg": f"调用 LLM 失败：{type(e).__name__}: {e}"})
            break

        out = _truncate_extra(out)
        messages.append({"role": "assistant", "content": out})

        # 展示 Thought
        mt = re.search(r"Thought:\s*(.*?)(?=\n\s*Action:|\Z)", out, re.DOTALL)
        if mt:
            emit({"type": "thought", "text": mt.group(1).strip()[:600]})

        kind, payload = _parse_action(out)

        if kind == "finish":
            final_answer = payload
            emit({"type": "finish", "text": final_answer})
            break
        if kind != "tool":
            # ⚠️ 模型偶尔只输出 Thought 忘了 Action（实测出现过）。
            #    这里给一个更强的纠正提示，避免白白浪费一轮。
            obs = ("错误：你的回复里没有 Action 字段。\n"
                   "必须严格按这个格式回复，且 Action 只能是一行：\n"
                   "Thought: 你的思考\n"
                   "Action: 工具名()   或   Action: 工具名(arg=\"值\")\n"
                   "可用工具：classify_local() / analyze_vision() / "
                   "search_specs(query=\"车型名\")\n"
                   "现在就只输出这一对 Thought-Action，不要输出其它内容。")
            emit({"type": "observation", "tool": "(格式错误，已要求重试)", "text": obs})
            messages.append({"role": "user", "content": f"Observation: {obs}"})
            continue

        # 解析 function_name(arg="value")
        # ⚠️ payload 已剥离了 "Action:" 前缀（见 _parse_action），
        #    所以直接找函数名即可，不能再去找 "Action:"。
        mfn = re.search(r"(\w+)\s*\(", payload)
        if not mfn:
            obs = "错误：Action 里没有找到工具名。"
            emit({"type": "observation", "tool": "(无)", "text": obs})
            messages.append({"role": "user", "content": f"Observation: {obs}"})
            continue
        name = mfn.group(1)

        if name not in tools.TOOLS:
            obs = f"错误：未定义的工具 '{name}'。可用：{list(tools.TOOLS)}"
            emit({"type": "observation", "tool": name, "text": obs})
            messages.append({"role": "user", "content": f"Observation: {obs}"})
            continue

        # 参数
        margs = re.search(r"\((.*)\)", payload, re.DOTALL)
        kwargs = {}
        if margs:
            kwargs = {k: v for k, v in
                      re.findall(r'(\w+)\s*=\s*"([^"]*)"', margs.group(1))}

        # 注入图片（工具需要）
        fn = tools.TOOLS[name]
        if name in ("classify_local", "analyze_vision"):
            kwargs["image_bytes"] = ctx["image"]

        emit({"type": "action", "tool": name, "args": kwargs.get("query", "")})
        try:
            obs = fn(**kwargs)
        except Exception as e:
            obs = f"错误：工具执行异常 - {type(e).__name__}: {e}"

        emit({"type": "observation", "tool": name, "text": obs})
        messages.append({"role": "user", "content": f"Observation: {obs}"})

    if not final_answer:
        # 超出轮次仍未收尾：直接取最后一次观察做个兜底总结
        last_obs = ""
        for s in reversed(steps):
            if s["type"] == "observation":
                last_obs = s["text"]
                break
        final_answer = ("（Agent 达到最大轮次，以下是已获得的信息）\n\n" + last_obs
                        if last_obs else "Agent 未能给出结论。")

    return {"answer": final_answer, "steps": steps,
            "elapsed": round(time.time() - t0, 2)}
