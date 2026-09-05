"""GeoAgent 核心：一个带工具调用循环的对话 agent。

设计要点：
- 模型无关：任何 OpenAI 兼容接口（/v1/chat/completions）都能用，
  通过环境变量 GEOAGENT_API_KEY / GEOAGENT_BASE_URL / GEOAGENT_MODEL 配置。
- 未配置 API key 时进入 offline 模式：不调用模型，仅暴露工具执行能力，
  方便先开发和测试领域工具本身。
- 工具循环最多 N 轮，防止失控。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .tools.base import ToolRegistry, registry as default_registry


class GeoAgent:
    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        max_tool_rounds: int = 6,
        system_prompt: str = (
            "你是地球物理与机器学习领域的研究助手。你可以调用工具读取地震/测井数据、"
            "做信号处理和快速建模。回答要严谨：引用工具返回的数值，不确定时明确说明。"
        ),
    ) -> None:
        self.registry = registry or default_registry
        self.max_tool_rounds = max_tool_rounds
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []
        self._client = None
        self._init_llm()

    # -- LLM 连接 -----------------------------------------------------------

    def _init_llm(self) -> None:
        api_key = os.environ.get("GEOAGENT_API_KEY")
        if not api_key:
            return
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "已设置 GEOAGENT_API_KEY 但未安装 openai 库：pip install openai"
            )
        self._client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("GEOAGENT_BASE_URL", "https://api.openai.com/v1"),
        )
        self._model = os.environ.get("GEOAGENT_MODEL", "gpt-4o-mini")

    @property
    def online(self) -> bool:
        return self._client is not None

    # -- 工具执行（两种模式都可用） -------------------------------------------

    def run_tool(self, name: str, args: Dict[str, Any]) -> str:
        return self.registry.execute(name, args)

    # -- 对话 ----------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """发送一条消息；在线模式下自动执行模型请求的工具并把结果回传。"""
        self.history.append({"role": "user", "content": user_message})
        if not self.online:
            reply = (
                "[offline 模式] 未配置 GEOAGENT_API_KEY，无法调用 LLM。\n"
                f"可用工具: {[t.name for t in self.registry.list()]}\n"
                "可先离线开发/测试工具；或设置环境变量后重试。"
            )
            self.history.append({"role": "assistant", "content": reply})
            return reply

        messages = [{"role": "system", "content": self.system_prompt}, *self.history]
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
            resp = self._client.chat.completions.create(
                model=self._model, messages=messages, tools=tools
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                self.history.append({"role": "assistant", "content": msg.content or ""})
                return msg.content or ""
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        tc.model_dump() for tc in msg.tool_calls
                    ],
                }
            )
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                result = self.registry.execute(tc.function.name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )
        return "已达单轮工具调用上限，请拆分任务后重试。"
