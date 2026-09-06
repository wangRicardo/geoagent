"""Ricardo Agent 核心：流式工具循环 + 事件回调 + 会话快照。

设计要点：
- 模型无关：任何 OpenAI 兼容接口都能用；提供商/模型/思考强度热切换并持久化。
- 流式：模型回复逐字推送（delta 事件），工具调用过程有 start/end 事件，
  CLI / GUI 通过 on_event 回调实时呈现，界面不再是黑盒。
- 韧性：网络/API 错误自动指数退避重试（最多 3 次，未收到任何输出前才重试）。
- 上下文：历史自动裁剪（max_history），防止长对话撑爆上下文。
- 工作区：agent 在指定文件夹内工作，文件工具限制在目录内。
- 会话：save/load 快照到 ~/.geoagent/sessions/，跨天继续课题。
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import AgentConfig
from .tools.base import ToolRegistry
from .tools.base import registry as default_registry

SESSIONS_DIR = Path.home() / ".geoagent" / "sessions"
EventCallback = Callable[[dict[str, Any]], None] | None


class GeoAgent:
    def __init__(
        self,
        registry: ToolRegistry | None = None,
        max_tool_rounds: int = 8,
        max_history: int = 40,
        workdir: str | None = None,
        system_prompt: str = (
            "你是 Ricardo Agent，地球物理与机器学习领域的研究助手。你在用户指定的"
            "工作文件夹内工作，可以调用工具读写文件、读取地震/测井数据、做信号处理、"
            "运行 Python 代码和快速建模。回答要严谨：引用工具返回的数值，不确定时明确说明。"
            "用 plot_* 工具生成图件后，应调用 see_image 做视觉自查（检查标签、配色、数据异常），"
            "发现问题就修正数据或参数重新绘图，直到合格再交付。"
        ),
    ) -> None:
        self.registry = registry or default_registry
        self.max_tool_rounds = max_tool_rounds
        self.max_history = max_history
        self.system_prompt = system_prompt
        self.config = AgentConfig()
        self.history: list[dict[str, Any]] = []
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
        self._client = None
        self._client_key: tuple = ()
        self.workdir = os.path.abspath(workdir or os.getcwd())
        if workdir:
            os.makedirs(self.workdir, exist_ok=True)
        os.chdir(self.workdir)
        self._init_llm()

    # -- 工作目录 ---------------------------------------------------------------

    def set_workdir(self, path: str) -> str:
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(path):
            return f"ERROR: 目录不存在: {path}"
        self.workdir = path
        os.chdir(path)
        self.history = []
        return f"工作目录已切换: {path}"

    # -- LLM 连接 ---------------------------------------------------------------

    def _init_llm(self, force: bool = False) -> None:
        key = self.config.api_key
        if not key:
            self._client = None
            self._client_key = ()
            return
        cfg_key = (self.config.provider, self.config.model, self.config.thinking, self.config.base_url)
        if not force and cfg_key == self._client_key and self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("需要 openai 库：pip install openai") from exc
        self._client = OpenAI(api_key=key, base_url=self.config.base_url)
        self._client_key = cfg_key

    @property
    def online(self) -> bool:
        self._init_llm()
        return self._client is not None

    def reload(self) -> str:
        self._init_llm(force=True)
        self.history = []
        return f"已重载配置并清空对话历史。\n{self.config.status()}"

    # -- 带重试的流式请求 -----------------------------------------------------------

    def _create_stream(self, params: dict[str, Any]):
        """指数退避重试（最多 3 次）；流开始后不再重试，避免重复输出。"""
        last_exc: Exception | None = None
        # 请求用量统计；不支持的端点会报错，去掉该参数重试一次
        params = dict(params, stream_options={"include_usage": True})
        for attempt in range(3):
            try:
                return self._client.chat.completions.create(**params, stream=True)
            except Exception as exc:  # noqa: BLE001
                if (
                    "stream_options" in params
                    and attempt == 0
                    and ("stream_options" in str(exc) or "usage" in str(exc).lower())
                ):
                    params.pop("stream_options")
                    continue
                last_exc = exc
                if attempt < 2:
                    time.sleep(1.5 * (2**attempt))
        raise RuntimeError(f"LLM 调用失败（已重试 3 次）: {last_exc}")

    def _consume_stream(self, stream, emit: Callable[[dict[str, Any]], None]):
        """消费流式响应：文本逐字 emit；工具调用按 index 拼装；统计用量。"""
        parts: list[str] = []
        tc: dict[int, dict[str, str]] = {}
        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:  # 最后一个 chunk 携带用量
                self.usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
                self.usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
                self.usage["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
                self.usage["calls"] += 1
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            if delta.content:
                parts.append(delta.content)
                emit({"type": "delta", "text": delta.content})
            for d in delta.tool_calls or []:
                slot = tc.setdefault(d.index, {"id": "", "name": "", "args": ""})
                if d.id:
                    slot["id"] = d.id
                if d.function:
                    if d.function.name:
                        slot["name"] += d.function.name
                    if d.function.arguments:
                        slot["args"] += d.function.arguments
        return "".join(parts), tc

    # -- 上下文与请求参数 ------------------------------------------------------------

    def _trim_history(self) -> None:
        if len(self.history) <= self.max_history:
            return
        drop = len(self.history) - self.max_history
        drop += drop % 2  # 成对丢，保证剩余首条是 user
        self.history = self.history[drop:]

    def _request_params(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        params: dict[str, Any] = dict(model=self.config.model, messages=messages)
        if tools:
            params["tools"] = tools
        if self.config.thinking != "off":
            params["reasoning_effort"] = self.config.thinking
        return params

    # -- 会话快照 ----------------------------------------------------------------

    def usage_report(self) -> str:
        u = self.usage
        return (
            f"本次会话 LLM 用量: 调用 {u['calls']} 次 | "
            f"输入 {u['prompt_tokens']:,} + 输出 {u['completion_tokens']:,} "
            f"= {u['total_tokens']:,} tokens"
        )

    def save_session(self, name: str = "session") -> str:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        safe = "".join(c for c in name if c.isalnum() or c in "-_") or "session"
        path = SESSIONS_DIR / f"{safe}.json"
        path.write_text(
            json.dumps(
                {
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "workdir": self.workdir,
                    "provider": self.config.provider,
                    "model": self.config.model,
                    "thinking": self.config.thinking,
                    "history": self.history,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return f"会话已保存: {path.name}（{len(self.history)} 条消息）"

    def load_session(self, name: str = "session") -> str:
        safe = "".join(c for c in name if c.isalnum() or c in "-_") or "session"
        path = SESSIONS_DIR / f"{safe}.json"
        if not path.exists():
            return f"ERROR: 会话 {name!r} 不存在。可用: {self.list_sessions()}"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.config.set_provider(data["provider"])
        self.config.set_model(data["model"])
        self.config.set_thinking(data["thinking"])
        self.set_workdir(data["workdir"])
        self.history = data["history"]
        return (
            f"会话已恢复: {path.name}（{data['saved_at']} 保存，{len(self.history)} 条消息，"
            f"工作区 {self.workdir}）"
        )

    def list_sessions(self) -> str:
        if not SESSIONS_DIR.exists():
            return "（还没有保存的会话）"
        files = sorted(SESSIONS_DIR.glob("*.json"))
        return ", ".join(p.stem for p in files) if files else "（还没有保存的会话）"

    def export_transcript(self, path: str) -> str:
        """把当前对话导出为 Markdown 研究日志（写到工作目录内）。"""
        import re

        root = os.getcwd()
        full = os.path.abspath(os.path.join(root, path))
        if os.path.commonpath([root, full]) != root:
            return f"ERROR: 路径越出工作目录: {path}"
        lines = [
            "# Ricardo Agent 对话记录",
            "",
            f"- 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- 工作区: `{self.workdir}`",
            f"- 模型: {self.config.provider}/{self.config.model} (思考: {self.config.thinking})",
            "",
        ]
        for m in self.history:
            role = {"user": "## 🧑 你", "assistant": "## 🤖 Ricardo"}.get(m["role"], f"## {m['role']}")
            content = m["content"]
            if m["role"] == "user":
                content = re.sub(r"\[当前工作目录[^\]]*\]\s*", "", content)
            lines += [role, "", content, ""]
        with open(full, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return f"对话已导出: {full}（{len(self.history)} 条消息）"

    # -- 工具执行 ---------------------------------------------------------------

    def run_tool(self, name: str, args: dict[str, Any]) -> str:
        return self.registry.execute(name, args)

    # -- 对话 ----------------------------------------------------------------

    def chat(self, user_message: str, on_event: EventCallback = None) -> str:
        """发送一条消息。on_event 收到实时事件：
        {"type": "delta", "text": ...} / {"type": "tool_start", "name","args"}
        / {"type": "tool_end", "name", "result"} / {"type": "error", "message"}
        """
        emit: Callable[[dict[str, Any]], None] = on_event or (lambda ev: None)
        self.history.append({"role": "user", "content": user_message})
        self._trim_history()
        if not self.online:
            reply = (
                "[offline 模式] 未检测到 API 密钥。请设置提供商对应的环境变量"
                f"（或通用的 RICARDO_API_KEY），对话中可用 /provider 查看当前提供商。\n"
                f"可用工具: {[t.name for t in self.registry.list()]}"
            )
            self.history.append({"role": "assistant", "content": reply})
            return reply

        messages = [
            {"role": "system", "content": self.system_prompt + f"\n[当前工作目录: {self.workdir}]"},
            *self.history,
        ]
        tools = [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
            }
            for t in self.registry.list()
        ]
        for _ in range(self.max_tool_rounds):
            params = self._request_params(messages, tools)
            stream = self._create_stream(params)
            content, tc_acc = self._consume_stream(stream, emit)
            if not tc_acc:
                self.history.append({"role": "assistant", "content": content})
                return content
            tool_calls = [
                {
                    "id": s["id"] or f"call_{i}",
                    "type": "function",
                    "function": {"name": s["name"], "arguments": s["args"]},
                }
                for i, s in sorted(tc_acc.items())
            ]
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                emit({"type": "tool_start", "name": name, "args": args})
                result = self.registry.execute(name, args)
                emit({"type": "tool_end", "name": name, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        return "已达单轮工具调用上限，请拆分任务后重试。"
