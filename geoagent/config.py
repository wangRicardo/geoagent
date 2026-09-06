"""Ricardo Agent 配置：LLM 提供商、模型、思考强度，持久化到 ~/.geoagent/config.json。

对话中可随时用 /provider /model /thinking 热切换，无需重启。
API 密钥不写入配置文件，仍从环境变量读取（按提供商约定命名，
如 OPENAI_API_KEY / DEEPSEEK_API_KEY，或通用的 RICARDO_API_KEY）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG_PATH = Path.home() / ".geoagent" / "config.json"

# 内置提供商预设：base_url 与 API key 的环境变量名。
PROVIDERS: dict[str, dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "o4-mini"],
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "key_env": "ZHIPU_API_KEY",
        "default_model": "glm-4-flash",
        "models": ["glm-4-flash", "glm-4-plus", "glm-4.5"],
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "key_env": "MOONSHOT_API_KEY",
        "default_model": "moonshot-v1-8k",
        "models": ["moonshot-v1-8k", "moonshot-v1-32k", "kimi-k2-0711-preview"],
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key_env": "DASHSCOPE_API_KEY",
        "default_model": "qwen-plus",
        "models": ["qwen-plus", "qwen-max", "qwen3-235b-a22b"],
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key_env": "OLLAMA_API_KEY",  # 本地 ollama 通常不需要真实 key
        "default_model": "qwen3:8b",
        "models": ["qwen3:8b", "llama3.1:8b"],
    },
}

THINKING_LEVELS = ["off", "low", "medium", "high"]


class AgentConfig:
    """当前会话配置：提供商 / 模型 / 思考强度 / API key，可热切换并持久化。"""

    def __init__(self) -> None:
        data = self._load()
        self.provider: str = data.get("provider", "openai")
        self.model: str = data.get("model", self._preset()["default_model"])
        self.thinking: str = data.get("thinking", "off")
        self._custom_models: list[str] = data.get("custom_models", [])

    # -- 持久化 ---------------------------------------------------------------

    def _load(self) -> dict:
        if CONFIG_PATH.exists():
            try:
                return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def save(self) -> None:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(
            json.dumps(
                {
                    "provider": self.provider,
                    "model": self.model,
                    "thinking": self.thinking,
                    "custom_models": self._custom_models,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # -- 提供商 ----------------------------------------------------------------

    def _preset(self) -> dict[str, str]:
        return PROVIDERS.get(self.provider, PROVIDERS["openai"])

    def set_provider(self, name: str) -> str:
        if name not in PROVIDERS:
            return f"ERROR: 未知提供商 {name!r}。可用: {', '.join(PROVIDERS)}"
        self.provider = name
        if self.model not in self.models():
            self.model = self._preset()["default_model"]
        self.save()
        return f"已切换到 {name}（模型: {self.model}）"

    @property
    def base_url(self) -> str:
        # 环境变量可覆盖预设，方便接私有网关。
        return os.environ.get("GEOAGENT_BASE_URL", self._preset()["base_url"])

    @property
    def api_key(self) -> str | None:
        key_env = self._preset()["key_env"]
        return (
            os.environ.get("RICARDO_API_KEY") or os.environ.get(key_env) or os.environ.get("GEOAGENT_API_KEY")
        )

    # -- 模型 ------------------------------------------------------------------

    def models(self) -> list[str]:
        return self._preset()["models"] + self._custom_models

    def set_model(self, name: str) -> str:
        if name not in self.models():
            self._custom_models.append(name)
        self.model = name
        self.save()
        return f"模型已切换为 {name}"

    # -- 思考强度 ----------------------------------------------------------------

    def set_thinking(self, level: str) -> str:
        if level not in THINKING_LEVELS:
            return f"ERROR: 思考强度可选 {', '.join(THINKING_LEVELS)}"
        self.thinking = level
        self.save()
        return f"思考强度已设为 {level}"

    def fetch_models(self) -> list[str] | None:
        """从提供商的 /models 端点拉取真实模型列表；失败返回 None（调用方回退硬编码）。"""
        import requests

        if not self.api_key:
            return None
        try:
            r = requests.get(
                self.base_url.rstrip("/") + "/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=6,
            )
            data = r.json().get("data", [])
            ids = sorted(m.get("id") for m in data if m.get("id"))
            return ids or None
        except Exception:  # noqa: BLE001
            return None

    def status(self) -> str:
        has_key = "已配置" if self.api_key else "未配置密钥"
        return (
            f"提供商: {self.provider} ({self.base_url})\n"
            f"模型: {self.model}\n"
            f"思考强度: {self.thinking} | 密钥: {has_key}"
        )
