"""Ricardo Agent 核心：带工具调用循环、可热切换模型配置的对话 agent。

设计要点：
- 模型无关：任何 OpenAI 兼容接口都能用；提供商/模型/思考强度可在对话中
  用 /provider /model /thinking 热切换（AgentConfig 持久化）。
- 工作目录：agent 在指定文件夹内工作（类似 codex / claude code），
  文件系统工具都相对于当前工作目录解析。
- 未配置任何密钥时进入 offline 模式，仅暴露工具执行能力。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .config import AgentConfig, THINKING_LEVELS
from .tools.base import ToolRegistry, registry as default_registry


class GeoAgent:
    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        max_tool_rounds: int = 8,
        workdir: Optional[str] = None,
        system_prompt: str = (
            "你是 Ricardo Agent，地球物理与机器学习领域的研究助手。你在用户指定的"
            "工作文件夹内工作，可以调用工具读写文件、读取地震/测井数据、做信号处理、"
            "运行 Python 代码和快速建模。回答要严谨：引用工具返回的数值，不确定时明确说明。"
        ),
    ) -> None:
        self.registry = registry or default_registry
        self.max_tool_rounds = max_tool_rounds
        self.system_prompt = system_prompt
        self.config = AgentConfig()
        self.history: List[Dict[str, Any]] = []
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
        self.history = []  # 换工作区后清空上下文，避免旧文件名串场
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
        except ImportError:
            raise RuntimeError("需要 openai 库：pip install openai")
        self._client = OpenAI(api_key=key, base_url=self.config.base_url)
        self._client_key = cfg_key

    @property
    def online(self) -> bool:
        self._init_llm()  # 密钥可能在运行中被设置，每次检查刷新
        return self._client is not None

    def reload(self) -> str:
        """配置变更后重建客户端。"""
        self._init_llm(force=True)
        self.history = []
        return f"已重载配置并清空对话历史。\n{self.config.status()}"

    # -- 工具执行 ---------------------------------------------------------------

    def run_tool(self, name: str, args: Dict[str, Any]) -> str:
        return self.registry.execute(name, args)

    # -- 对话 ----------------------------------------------------------------

    def _request_params(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        params: Dict[str, Any] = dict(model=self.config.model, messages=messages)
        if tools:
            params["tools"] = tools
        if self.config.thinking != "off":
            # OpenAI o 系列等支持 reasoning_effort；不支持的端点会忽略或报错，
            # 报错时由调用方去掉该参数重试。
            params["reasoning_effort"] = self.config.thinking
        return params

    def chat(self, user_message: str) -> str:
        """发送一条消息；在线模式下自动执行模型请求的工具并把结果回传。"""
        self.history.append({"role": "user", "content": user_message})
        if not self.online:
            reply = (
                "[offline 模式] 未检测到 API 密钥。请设置提供商对应的环境变量"
                f"（或通用的 RICARDO_API_KEY），对话中可用 /provider 查看当前提供商。\n"
                f"可用工具: {[t.name for t in self.registry.list()]}"
            )
            self.history.append({"role": "assistant", "content": reply})
            return reply

        workspace_note = f"[当前工作目录: {self.workdir}]"
        messages = [
            {"role": "system", "content": self.system_prompt + "\n" + workspace_note},
            *self.history,
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self.registry.list()
        ]
        for _ in range(self.max_tool_rounds):
            params = self._request_params(messages, tools)
            try:
                resp = self._client.chat.completions.create(**params)
            except Exception as exc:  # noqa: BLE001
                if "reasoning_effort" in params:
                    params.pop("reasoning_effort")
                    resp = self._client.chat.completions.create(**params)
                else:
                    raise RuntimeError(f"LLM 调用失败: {exc}") from exc
            msg = resp.choices[0].message
            if not msg.tool_calls:
                self.history.append({"role": "assistant", "content": msg.content or ""})
                return msg.content or ""
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                }
            )
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                result = self.registry.execute(tc.function.name, args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        return "已达单轮工具调用上限，请拆分任务后重试。"


# THINKING_LEVELS 供 CLI 引用
__all__ = ["GeoAgent", "THINKING_LEVELS"]
