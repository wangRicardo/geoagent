"""工具注册与描述机制。

所有工具都是普通 Python 函数，通过 ``@tool`` 装饰器注册。
函数签名的类型注解和 docstring 会自动转成 JSON Schema 风格的描述，
便于将来对接任意 LLM 的 function-calling 接口。
"""

from __future__ import annotations

import inspect
import json
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    category: str = "general"

    def schema(self) -> Dict[str, Any]:
        """返回可交给 LLM function-calling 的工具描述。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "category": self.category,
        }

    def __call__(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)


def _python_type_to_json(tp: Any) -> str:
    origin = typing.get_origin(tp)
    if origin in (list, List):
        return "array"
    if origin in (dict, Dict):
        return "object"
    if tp in (int,):
        return "integer"
    if tp in (float,):
        return "number"
    if tp in (bool,):
        return "boolean"
    return "string"


def _build_parameters(func: Callable) -> Dict[str, Any]:
    sig = inspect.signature(func)
    props: Dict[str, Any] = {}
    required: List[str] = []
    for pname, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty:
            required.append(pname)
        props[pname] = {"type": _python_type_to_json(param.annotation or str)}
        if param.default is not inspect.Parameter.empty and param.default is not None:
            props[pname]["default"] = param.default
    schema = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, name: str = "", category: str = "general") -> Callable:
        """``@registry.register()`` 装饰器，把函数注册为工具。"""

        def deco(func: Callable) -> Callable:
            tool_name = name or func.__name__
            doc = inspect.getdoc(func) or ""
            # 只取 docstring 第一段作为描述，避免把长说明塞进 schema。
            description = doc.split("\n\n")[0].strip()
            self._tools[tool_name] = Tool(
                name=tool_name,
                description=description,
                func=func,
                parameters=_build_parameters(func),
                category=category,
            )
            return func

        return deco

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"未知工具: {name}。可用: {sorted(self._tools)}")
        return self._tools[name]

    def list(self) -> List[Tool]:
        return sorted(self._tools.values(), key=lambda t: (t.category, t.name))

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.schema() for t in self.list()]

    def execute(self, name: str, args: Dict[str, Any]) -> str:
        """执行工具并把结果转成字符串（agent 循环里统一用文本回传 LLM）。"""
        try:
            result = self.get(name)(**args)
        except Exception as exc:  # noqa: BLE001 - 工具错误要回传给模型自行纠正
            return f"ERROR: {type(exc).__name__}: {exc}"
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except TypeError:
            return str(result)


registry = ToolRegistry()
