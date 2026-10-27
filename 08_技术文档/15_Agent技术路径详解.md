# 从最简 ReAct Agent 到网页车辆识别 Agent

> 本文讲清这套网页 Agent 的**技术原理**，以及**如何自己动手实现**。
> 目标是让读者读完能独立复现，不需要额外依赖。

---

## 一、起点：一个最简的 ReAct Agent

一个最简的 ReAct Agent 通常只有百行左右：一个循环，加两三个示例工具（教材里常用天气查询）。
它虽然简单，但**已经包含了 ReAct 最核心的部分**：

```
用户提问 → LLM 思考 → 选择工具 → 执行 → 拿到结果 → 塞回历史 → 循环 → 结束
```

**它只能算是一个骨架，缺三样东西**：

| 缺什么 | 最简 ReAct Agent | 车辆识别 Agent |
|---|---|---|
| **① 输入** | 写死的字符串 `"北京今天天气"` | **用户上传图片** |
| **② 工具** | `get_weather` / `get_attraction` | 本地模型 / 云端视觉 / 搜索 |
| **③ 界面** | `print()` 打印到命令行 | **网页** |

**所以这不是重写，是"换零件"。** 下面逐个讲。

---

## 二、三个要换的零件，各自的技术点

### 零件 ① 输入：从"写死的字符串"到"用户上传图片"

#### 技术点 1：`input()` 不够用 —— 图片是二进制

命令行只能输入文字。图片是**二进制数据**，需要一条通道把它送到 Python。

**网页上传的本质就三行**：

```html
<input type="file" id="f">
<script>
const file = document.getElementById('f').files[0];   // 拿到文件
fetch('/api/analyze', {method:'POST', body: formData}); // 发给服务器
</script>
```

```python
# 服务器端从请求里取出图片字节
image_bytes = fields["image"]        # 就是二进制数据
```

#### 技术点 2：图片怎么传给 LLM？—— **这是最关键的设计决策**

有两种做法：

| 做法 | 说明 | 问题 |
|---|---|---|
| **A. 图片塞进对话历史** | 每轮 message 里都带图片 | ❌ ReAct 要循环 4 轮 → **图片传 4 次**，浪费 token，格式还容易出错 |
| **B. 图片由"工具"持有** ✅ | 图片只存在 Python 变量里，工具返回**纯文本**给 LLM | ✅ 省 token、格式简单、逻辑清晰 |

**本项目选择 B。** 代码结构如下：

```python
ctx = {"image": image_bytes}          # 图片存在这里，不进 LLM 对话

if name == "classify_local":
    kwargs["image_bytes"] = ctx["image"]   # 工具需要时注入

obs = fn(**kwargs)                     # 工具返回的是「纯文本」
messages.append({"role":"user", "content": f"Observation: {obs}"})
```

**LLM 全程只读文字，从没见过图片** —— 它只管"决策下一步调用哪个工具"。

> 💡 这个设计很重要，是"能用"和"好用"的分界线。

---

### 零件 ② 工具：三个工具，三种技术

#### 工具 1：本地深度学习模型（`classify_local`）

**技术点：ONNX Runtime 推理**

```python
import onnxruntime as ort
from PIL import Image
import numpy as np

sess = ort.InferenceSession("cars1000_resnet50_224.onnx",
                            providers=["CPUExecutionProvider"])

# 预处理必须与训练完全一致！顺序不能错
im = Image.open(path).convert("RGB").resize((224,224))
x = np.asarray(im, dtype=np.float32) / 255.0        # ① 归一化到 0~1
x = (x - [0.485,0.456,0.406]) / [0.229,0.224,0.225] # ② ImageNet 标准化
x = x.transpose(2,0,1)[None]                        # ③ HWC→NCHW，加 batch 维

out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
```

**⚠️ 两个必须注意的问题**：

| 坑 | 后果 |
|---|---|
| 预处理顺序/参数不一致 | 结果全错（**编译不报错、运行不报错**） |
| 类别索引顺序用了 JSON 而不是目录顺序 | **推理正常，但标签整体错位** |

**为什么它快**：模型只有 25.87 MB（INT8 量化），CPU 推理 **17 ms**。

#### 工具 2：云端多模态大模型（`analyze_vision`）

**技术点：OpenAI 兼容接口 + 多模态消息格式**

```python
from openai import OpenAI
cli = OpenAI(api_key=KEY, base_url="https://api.deepseek.com")   # 兼容接口！

import base64
b64 = base64.b64encode(open("car.jpg","rb").read()).decode()

resp = cli.chat.completions.create(
    model="deepseek-flash",
    messages=[{
        "role": "user",
        "content": [                                    # ← 变成「块数组」
            {"type": "text", "text": "这张图里是什么车？"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},   # ← 图片块
        ],
    }],
    max_tokens=1500,        # ⚠️ 见下方坑
)
```

**这里是最简 ReAct Agent 与车辆识别 Agent 最大的架构差异**：

| | 最简 ReAct Agent | 车辆识别 Agent |
|---|---|---|
| `content` 类型 | **字符串** | **块数组**（text 块 + image 块） |
| 图片怎么传 | — | **Base64** 内联（也可用 URL / Files API） |

**⚠️ 实现时容易遇到的问题**：

> `max_tokens` 设太小 → 模型把预算全花在**内部思考**上 → **正文还没输出就被截断，返回空字符串**。
>
> 最初的判断容易被误导为"模型不支持视觉"，实际只是参数问题。
> **判断方法：看 `finish_reason`。`length` = 被截断，不是不支持。**

#### 工具 3：联网搜索（`search_specs`）

**技术点：调用第三方搜索 API**

```python
from tavily import TavilyClient
tv = TavilyClient(api_key=KEY)
r = tv.search(query="宝马M5 参数 配置 价格",
              search_depth="basic", include_answer=True, max_results=3)
```

**为什么需要它**：LLM 的知识**冻结在训练时刻**，它不知道当前车价。
**这就是"给大模型配上手足"的含义** —— 用工具**取回实时信息**。

#### 工具的统一约定（很重要）

**每个工具都必须**：
1. 输入输出都是**简单类型**（bytes / str）
2. **出错时返回错误文本，而不是抛异常** ← 否则 Agent 循环会崩

```python
try:
    ...
except Exception as e:
    return f"错误：xxx 失败 - {type(e).__name__}: {e}"   # 返回，不抛
```

---

### 零件 ③ 界面：从 `print()` 到网页

#### 技术点 1：用 Python 标准库起一个网页服务

**不需要 flask！** Python 自带 `http.server`：

```python
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):        # 处理 GET：返回网页
        ...
    def do_POST(self):       # 处理 POST：接收图片
        ...

srv = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
srv.serve_forever()
```

**`ThreadingHTTPServer` 的 `Threading` 很关键** —— 每个请求一个线程，否则流式响应会阻塞其他请求。

#### 技术点 2：接收图片要做 multipart 解析

浏览器上传图片用的是 `multipart/form-data` 格式。要手动解析：

```python
def parse_multipart(body: bytes, boundary: bytes) -> dict:
    for part in body.split(b"--" + boundary):
        head, _, data = part.partition(b"\r\n\r\n")
        name = re.search(rb'name="([^"]*)"', head).group(1)
        out[name.decode()] = data          # 文件就是二进制
    return out
```

#### 技术点 3：SSE —— 让网页实时显示"思考过程" ⭐

**这是体验上最关键的一环。** 普通请求要等 27 秒才返回，页面一直白着；SSE 能**边跑边显示**。

**原理**：服务器返回一个"持续不断的流"，每条消息以 `data: ` 开头、`\n\n` 结尾：

```python
self.send_header("Content-Type", "text/event-stream")   # 关键 MIME
...
def push(ev):
    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
    self.wfile.flush()          # ⚠️ 必须 flush，否则攒着不发
```

**前端用 `fetch` + `ReadableStream` 读**：

```javascript
const reader = (await fetch('/api/analyze', {method:'POST', body:fd})).body.getReader();
while (true) {
  const {done, value} = await reader.read();
  if (done) break;
  buf += new TextDecoder().decode(value, {stream:true});
  // 按 \n\n 切分，逐条 data: 解析 → 更新页面
}
```

> **⚠️ 实际踩到的坑（表现为页面卡在"分析中"）**：
>
> SSE 是"长度不确定的持续流"。最初使用 `HTTP/1.1 + Connection: keep-alive`，
> 但**既没给 `Content-Length`，也没用 chunked 编码** →
> 浏览器**一直等连接结束**，页面永远停在"分析中"。
>
> **正确做法**：
> ```python
> self.protocol_version = "HTTP/1.0"      # 或
> self.send_header("Connection", "close")
> self.close_connection = True
> ```
> **判断技巧**：用 `curl -N` 测。如果 Agent 25 秒跑完、但 curl 一直挂着不退出，就是这个坑。

---

## 三、把它们拼起来：完整数据流

```
浏览器                      Python 服务器                   外部
  │                              │                          │
  │ ① 上传图片 (multipart)        │                          │
  ├─────────────────────────────▶│                          │
  │                              │ ② 解析出 image_bytes      │
  │                              │                          │
  │                              │ ③ 进入 ReAct 循环 ────────┤
  │ ④ SSE 实时推送 ◀──────────────┤   Thought/Action/Obs      │
  │    · round                   │        │                 │
  │    · thought                 │        ├─ classify_local ─┼─▶ ONNX 本地(17ms)
  │    · action                  │        ├─ analyze_vision ┼─▶ DeepSeek 云端(5s)
  │    · observation             │        └─ search_specs ──┼─▶ Tavily(5s)
  │    · finish                  │                          │
  │ ⑤ 显示最终结论 ◀──────────────┤ ⑥ Finish[答案]           │
```

---

## 四、实现步骤：从命令行 Agent 改造的分步清单

| 步骤 | 做什么 | 难点 |
|---|---|---|
| **1** | 复用已有的 ReAct 循环骨架 | 无（循环部分可直接沿用） |
| **2** | 把系统提示词改成车辆识别场景 | 让 LLM 知道先跑两个识别工具做对比 |
| **3** | 写 `classify_local(image_bytes)` | **预处理必须与训练一致** ⚠️ |
| **4** | 写 `analyze_vision(image_bytes)` | `content` 要改成**块数组**；`max_tokens` 给足 |
| **5** | 写 `search_specs(query)` | 无 |
| **6** | 图片用 `ctx` 变量持有，**不进对话** | 设计决策，很重要 |
| **7** | 起 `http.server`，两个路由 | `ThreadingHTTPServer` |
| **8** | 写 `index.html`（上传 + 步骤流） | 无 |
| **9** | 接 SSE，让步骤实时显示 | **HTTP/1.0 + close** ⚠️ |
| **10** | 写 `.bat` 一键启动 | 无 |

**最容易出错的是 3、4、9 这三步。**

---

## 五、值得记住的 6 条经验

| # | 经验 | 代价 |
|---|---|---|
| 1 | **预处理必须与训练完全一致** | 顺序/参数错 = 结果全错，且不报错 |
| 2 | **图片别进 LLM 对话上下文**，让工具持有 | 否则循环几轮就传几次，浪费又易错 |
| 3 | **`max_tokens` 要给足**（模型带思考链） | 太小 → 返回空字符串，像"不支持" |
| 4 | **SSE 必须用 HTTP/1.0 或显式 close** | 否则浏览器永远等，页面卡住 |
| 5 | **工具出错要返回文本，不要抛异常** | 否则 Agent 循环直接崩 |
| 6 | **密钥放 config.ini，不要硬编码** | 打包分发 = 密钥跟着走 |

---

## 六、这套架构的可复用性

**它与最简 ReAct Agent 是同一个骨架**，换掉的只有"工具"和"输入入口"：

| 场景 | 输入 | 工具 |
|---|---|---|
| 天气助手 | 文字提问 | 天气 API / 景点搜索 |
| **车辆识别** | **图片上传** | **本地模型 / 云端视觉 / 车辆搜索** |
| 以后的任何场景 | ? | ? |

> **ReAct 骨架不变，只换零件。**
> 这也说明：**ReAct 的骨架一旦搭好，就可以迁移到完全不同的任务上。**
> 真正需要针对场景重做的，只是"工具"和"输入入口"。
