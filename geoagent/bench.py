"""Agent 自评测基准：一组标准科研任务，检查期望的工具是否被正确调用。

防止改核心把 agent 行为改坏——每次改完 core/system prompt 跑一遍：
    ricardo bench            # 需要 API 密钥（在线运行）
    ricardo bench --list     # 只列出场景
场景定义在 benchmarks/scenarios.json，欢迎按自己的研究场景扩充。

通过标准：agent 的回复过程中调用了场景 expect_tools 里的全部工具
（顺序不限，允许调用额外工具）。
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable

from .tools.base import registry

SCENARIOS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmarks", "scenarios.json")


def load_scenarios(path: str | None = None) -> list[dict]:
    path = path or SCENARIOS_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)["scenarios"]


def run_benchmark(
    agent,
    scenarios: list[dict] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> str:
    """运行全部场景，返回 Markdown 评测报告。需要 agent 在线（会真实调用 LLM）。"""
    if not agent.online:
        return "ERROR: 评测需要 API 密钥（会真实调用 LLM）。请先配置密钥再运行 ricardo bench。"
    scenarios = scenarios or load_scenarios()
    log = (lambda m: on_progress(m)) if on_progress else (lambda m: None)
    results = []
    for sc in scenarios:
        name, expected = sc["name"], sc.get("expect_tools", [])
        unknown = [t for t in expected if t not in {x.name for x in registry.list()}]
        if unknown:
            results.append((name, False, f"场景引用了不存在的工具: {unknown}"))
            continue
        called: list[str] = []

        def on_event(ev, _called=called):
            if ev.get("type") == "tool_start":
                _called.append(ev.get("name", ""))

        log(f"▶ {name}: {sc.get('description', '')}")
        try:
            agent.chat(sc["user"], on_event=on_event)
        except Exception as exc:  # noqa: BLE001
            results.append((name, False, f"agent 抛出异常: {exc}"))
            continue
        missing = [t for t in expected if t not in called]
        if missing:
            results.append((name, False, f"缺少工具调用 {missing}，实际调用: {called or '无'}"))
        else:
            results.append((name, True, f"命中 {expected}，实际调用: {called}"))

    passed = sum(1 for _, ok, _ in results if ok)
    lines = [
        "# Ricardo Agent 评测报告",
        "",
        f"- 场景: {len(results)} | 通过: {passed} | 失败: {len(results) - passed}",
        f"- 通过率: {passed / len(results) * 100:.0f}%",
        "",
    ]
    for name, ok, detail in results:
        mark = "✅" if ok else "❌"
        lines.append(f"- {mark} **{name}** — {detail}")
    return "\n".join(lines)
