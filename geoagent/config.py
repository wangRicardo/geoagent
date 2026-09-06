"""Ricardo Agent 配置：LLM 提供商、模型、思考强度、权限模式，持久化到 ~/.geoagent/config.json。

- 对话中可随时用 /provider /model /thinking /mode 热切换，无需重启。
- 支持自定义厂商（base_url + 密钥 + 默认模型），在软件内添加、删除、切换。
- 密钥解析顺序：软件内保存的密钥（config.json）→ 环境变量（按厂商命名或通用
  RICARDO_API_KEY / GEOAGENT_API_KEY）。注意：config.json 中的密钥为明文本地存储，
  与 VS Code 等工具的做法一致，请勿与他人共享该文件。
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
PERMISSION_MODES = ["readonly", "standard", "ask", "full"]
PERMISSION_LABELS = {
    "readonly": "只读（仅检索/分析，禁止写入执行联网）",
    "standard": "标准（工作区内读写、可执行与联网）",
    "ask": "谨慎（写入/执行/联网逐次确认）",
    "full": "自主（全部允许，不再确认）",
}


class AgentConfig:
    """当前会话配置：提供商 / 模型 / 思考强度 / 权限模式 / API key，可热切换并持久化。"""

    def __init__(self) -> None:
        data = self._load()
        # 自定义厂商: {name: {base_url, key, default_model, models?}}
        self.custom_providers: dict[str, dict] = data.get("custom_providers", {})
        # 内置厂商的软件内密钥: {provider: key}
        self.saved_keys: dict[str, str] = data.get("keys", {})
        self.provider: str = data.get("provider", "openai")
        if self.provider not in PROVIDERS and self.provider not in self.custom_providers:
            self.provider = "openai"
        self.model: str = data.get("model", self._preset()["default_model"])
        self.thinking: str = data.get("thinking", "off")
        self.permission_mode: str = data.get("permission_mode", "standard")
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
                    "permission_mode": self.permission_mode,
                    "custom_models": self._custom_models,
                    "custom_providers": self.custom_providers,
                    "keys": self.saved_keys,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # -- 厂商解析 ----------------------------------------------------------------

    def _preset(self) -> dict:
        if self.provider in self.custom_providers:
            return self.custom_providers[self.provider]
        return PROVIDERS.get(self.provider, PROVIDERS["openai"])

    def provider_names(self) -> list[str]:
        """内置 + 自定义的全部厂商名。"""
        return list(PROVIDERS) + [n for n in self.custom_providers if n not in PROVIDERS]

    def upsert_provider(self, name: str, base_url: str, key: str, default_model: str) -> str:
        """添加或更新厂商（可覆盖内置厂商的地址与密钥）。"""
        name = name.strip().lower().replace(" ", "-")
        if not name or not base_url.startswith("http"):
            return "ERROR: 厂商名不能为空且 base_url 必须以 http 开头"
        self.custom_providers[name] = {
            "base_url": base_url.rstrip("/"),
            "key": key.strip(),
            "default_model": default_model.strip() or "gpt-3.5-turbo",
        }
        self.save()
        return f"厂商 {name} 已保存。可在提供商下拉中切换使用。"

    def delete_provider(self, name: str) -> str:
        if name not in self.custom_providers:
            return f"ERROR: {name!r} 不是自定义厂商（内置厂商不可删除）"
        del self.custom_providers[name]
        self.saved_keys.pop(name, None)
        if self.provider == name:
            self.provider = "openai"
            self.model = self._preset()["default_model"]
        self.save()
        return f"自定义厂商 {name} 已删除。"

    def set_key(self, provider: str, key: str) -> str:
        """为某厂商保存软件内密钥（优先级高于环境变量）。"""
        if provider not in self.provider_names():
            return f"ERROR: 未知厂商 {provider!r}"
        self.saved_keys[provider] = key.strip()
        self.save()
        return f"已保存 {provider} 的密钥（本地明文存储，勿外传 config.json）"

    def remove_key(self, provider: str) -> str:
        if self.saved_keys.pop(provider, None) is not None:
            self.save()
            return f"已清除 {provider} 的软件内密钥（环境变量仍生效）。"
        return f"{provider} 没有保存过软件内密钥。"

    # -- 提供商切换 ----------------------------------------------------------------

    def set_provider(self, name: str) -> str:
        if name not in self.provider_names():
            return f"ERROR: 未知提供商 {name!r}。可用: {', '.join(self.provider_names())}"
        self.provider = name
        if self.model not in self.models():
            self.model = self._preset()["default_model"]
        self.save()
        source = "自定义" if name in self.custom_providers else "内置"
        return f"已切换到 {name}（{source}，模型: {self.model}）"

    @property
    def base_url(self) -> str:
        # 环境变量可覆盖预设，方便接私有网关。
        return os.environ.get("GEOAGENT_BASE_URL", self._preset()["base_url"])

    @property
    def api_key(self) -> str | None:
        """密钥解析：软件内保存 → 通用环境变量 → 厂商专属环境变量。"""
        key = self.saved_keys.get(self.provider) or self.custom_providers.get(self.provider, {}).get("key")
        if key:
            return key
        key_env = self._preset().get("key_env", "")
        return (
            os.environ.get("RICARDO_API_KEY")
            or (os.environ.get(key_env) if key_env else None)
            or os.environ.get("GEOAGENT_API_KEY")
        )

    def key_source(self) -> str:
        if self.saved_keys.get(self.provider) or self.custom_providers.get(self.provider, {}).get("key"):
            return "软件内"
        if self.api_key:
            return "环境变量"
        return "未配置"

    # -- 模型 ------------------------------------------------------------------

    def models(self) -> list[str]:
        own = self.custom_providers.get(self.provider, {}).get("default_model")
        base = self._preset().get("models", [])
        extra = [own] if own and own not in base else []
        return base + extra + self._custom_models

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

    # -- 权限模式 ----------------------------------------------------------------

    def set_permission_mode(self, mode: str) -> str:
        if mode not in PERMISSION_MODES:
            return f"ERROR: 权限模式可选 {', '.join(PERMISSION_MODES)}"
        self.permission_mode = mode
        self.save()
        return f"权限模式已设为 {mode}（{PERMISSION_LABELS[mode]}）"

    # -- 其他 ----------------------------------------------------------------

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
        return (
            f"提供商: {self.provider} ({self.base_url}) | 密钥来源: {self.key_source()}\n"
            f"模型: {self.model}\n"
            f"思考强度: {self.thinking} | 权限模式: {self.permission_mode}"
        )
