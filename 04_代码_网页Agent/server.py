# -*- coding: utf-8 -*-
r"""
车辆识别 Agent —— 网页服务（Python 标准库，无需额外依赖）

启动：
    python server.py
然后浏览器打开 http://127.0.0.1:8000

接口：
    GET  /                 首页
    POST /api/analyze      上传图片，SSE 流式返回 Agent 思考过程
    GET  /api/status       配置与依赖状态
"""
from __future__ import annotations

import json
import re
import socket
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config_loader as cfg        # noqa: E402
import agent_core                  # noqa: E402

INDEX_HTML = HERE / "static" / "index.html"


# ----------------------------------------------------------------------
# multipart/form-data 解析（只提取所需字段）
# ----------------------------------------------------------------------
def parse_multipart(body: bytes, boundary: bytes) -> dict:
    """返回 {字段名: bytes 或 str}"""
    out: dict = {}
    delim = b"--" + boundary
    for part in body.split(delim):
        if not part or part in (b"--\r\n", b"--", b"\r\n"):
            continue
        if b"\r\n\r\n" not in part:
            continue
        head, _, data = part.partition(b"\r\n\r\n")
        data = data.rstrip(b"\r\n")
        head_txt = head.decode("utf-8", "replace")
        m = re.search(r'name="([^"]*)"', head_txt)
        if not m:
            continue
        name = m.group(1)
        if "filename=" in head_txt:
            out[name] = data                 # 文件：保留 bytes
        else:
            out[name] = data.decode("utf-8", "replace")
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass                                  # 静音常规日志

    # ---------------- GET ----------------
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            if INDEX_HTML.exists():
                b = INDEX_HTML.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                # 禁止缓存，避免浏览器用旧的 JS 逻辑
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.end_headers()
                self.wfile.write(b)
            else:
                self._send(500, "text/plain; charset=utf-8",
                           "缺少 static/index.html".encode("utf-8"))
        elif self.path == "/api/status":
            st = {
                "llm": cfg.has_llm(),
                "search": cfg.has_search(),
                "model": cfg.ONNX_PATH.exists(),
                "labels": cfg.LABELS_CN.exists() or cfg.LABELS_EN.exists(),
                "model_name": cfg.ONNX_PATH.name,
                "deepseek_model": cfg.get("deepseek_model"),
            }
            self._send(200, "application/json; charset=utf-8",
                       json.dumps(st, ensure_ascii=False).encode("utf-8"))
        else:
            self._send(404, "text/plain; charset=utf-8", b"404")

    # ---------------- POST ----------------
    def do_POST(self):
        if self.path != "/api/analyze":
            self._send(404, "text/plain; charset=utf-8", b"404")
            return

        ctype = self.headers.get("Content-Type", "")
        m = re.search(r"boundary=([^;]+)", ctype)
        if not m:
            self._send(400, "text/plain; charset=utf-8",
                       "需要 multipart/form-data".encode("utf-8"))
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        fields = parse_multipart(body, m.group(1).strip('"').encode())

        image = fields.get("image")
        note = fields.get("note", "") or ""
        if isinstance(note, bytes):
            note = note.decode("utf-8", "replace")
        if not isinstance(image, (bytes, bytearray)) or not image:
            self._send(400, "text/plain; charset=utf-8",
                       "没有收到图片".encode("utf-8"))
            return

        # ---- SSE 流式响应 ----
        # ⚠️ 关键：SSE 是「持续流」，长度不确定，必须用 HTTP/1.0 + close
        #    否则在 HTTP/1.1 keep-alive 下浏览器会一直等，页面卡在"分析中"。
        self.protocol_version = "HTTP/1.0"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")     # 防反向代理缓冲
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        lock = threading.Lock()

        def push(ev: dict):
            try:
                with lock:
                    payload = "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
                    self.wfile.write(payload.encode("utf-8"))
                    self.wfile.flush()
            except Exception:
                pass

        # 立即推一个事件，避免浏览器等待
        push({"type": "open", "msg": "已连接"})

        try:
            result = agent_core.run_agent(image, note, on_event=push)
            push({"type": "done", "answer": result["answer"],
                  "elapsed": result["elapsed"]})
        except Exception as e:
            push({"type": "error", "msg": f"{type(e).__name__}: {e}"})
        finally:
            try:
                with lock:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
            except Exception:
                pass

    # ---------------- 工具 ----------------
    def _send(self, code, ctype, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _free_port(preferred: int) -> int:
    """端口被占用就往后找一个"""
    for p in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return preferred


def main() -> int:
    host = cfg.get("host") or "127.0.0.1"
    port = _free_port(int(cfg.get("port") or 8000))

    print("=" * 70)
    print("车辆识别 Agent —— ReAct + 本地模型 + 云端大模型")
    print("=" * 70)
    print(cfg.summary())
    print(f"  服务地址 : http://{host}:{port}")
    print("=" * 70)
    print("提示：按 Ctrl+C 停止服务\n")

    srv = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
