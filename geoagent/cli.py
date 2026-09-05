"""GeoAgent 命令行入口（作为包模块，支持 pip 安装后的 `geoagent` 命令）。"""

import json
import re
import sys

from . import GeoAgent, registry


def _demo_reply(line: str, agent: GeoAgent) -> str:
    """离线简单对话模式：关键词匹配直接调用工具，不依赖 LLM。"""
    low = line.lower().strip()

    def _run(name: str, args: dict) -> str:
        return agent.run_tool(name, args)

    if low in ("help", "帮助", "?"):
        return (
            "简单模式支持的指令（也支持自然关键词）：\n"
            "  工具列表                - 查看所有工具\n"
            "  ricker 主频=30          - 生成 Ricker 子波\n"
            "  调谐 主频=25 vp=3200    - 分辨率极限估算\n"
            "  时深 vp=3000 twt=[1000,2000] - TWT 转深度\n"
            "  记笔记 标题=... 内容=... [标签=a,b]\n"
            "  看笔记 [标签]\n"
            "  搜笔记 关键词\n"
            "接入真实 LLM 后可完全自然语言对话：设置 GEOAGENT_API_KEY 后重启。"
        )
    if "工具" in low or low == "tools":
        return "\n".join(f"[{t.category}] {t.name}: {t.description[:50]}" for t in registry.list())

    m = re.search(r"ricker|子波", low)
    if m:
        f = _num(line, "freq_hz", "主频", default=30.0)
        return _run("ricker_wavelet", {"freq_hz": f})
    if low.startswith(("搜笔记", "search_notes")):
        kw = low.replace("搜笔记", "").replace("search_notes", "").strip()
        return _run("search_notes", {"keyword": kw}) if kw else "请给出关键词，例如：搜笔记 调谐"
    if "调谐" in low or "分辨率" in low:
        return _run("tuning_thickness", {
            "freq_hz": _num(line, "freq_hz", "主频", default=25.0),
            "vp_ms": _num(line, "vp_ms", "vp", default=3000.0),
        })
    if "时深" in low or "深度转换" in low:
        twt = re.search(r"twt\s*=\s*(\[[^\]]*\])", low)
        if not twt:
            return "请给出时间列表，例如：时深 vp=3000 twt=[1000,2000]"
        return _run("velocity_to_depth", {
            "vp_ms": _num(line, "vp_ms", "vp", default=3000.0),
            "two_way_time_ms": json.loads(twt.group(1)),
        })
    if low.startswith(("记笔记", "save_note")):
        title = _kv(line, "标题") or "未命名笔记"
        content = _kv(line, "内容") or ""
        tags = _kv(line, "标签") or ""
        return _run("save_note", {"title": title, "content": content, "tags": tags})
    if low.startswith(("看笔记", "list_notes")):
        return _run("list_notes", {"tag": _kv(line, "标签") or ""})
    return (
        "（简单模式）我没听懂这句。输入 help 查看支持的指令；"
        "或配置 GEOAGENT_API_KEY 使用完整 LLM 对话。"
    )


def _num(line: str, eng: str, cn: str, default: float) -> float:
    m = re.search(rf"{eng}\s*=\s*([\d.]+)", line.lower()) or re.search(rf"{cn}\s*=\s*([\d.]+)", line)
    return float(m.group(1)) if m else default


def _kv(line: str, key: str) -> str:
    m = re.search(rf"{key}\s*=\s*(\S+)", line)
    return m.group(1) if m else ""


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tools"
    agent = GeoAgent()

    if cmd == "tools":
        for t in registry.list():
            print(f"[{t.category:10s}] {t.name}: {t.description[:60]}")
    elif cmd == "run":
        name = sys.argv[2]
        args: dict = {}
        if len(sys.argv) > 3:
            for pair in sys.argv[3].split(";"):
                pair = pair.strip()
                if not pair:
                    continue
                k, v = pair.split("=", 1)
                try:
                    v = json.loads(v)
                except json.JSONDecodeError:
                    pass
                args[k] = v
        print(agent.run_tool(name, args))
    elif cmd == "chat":
        demo = "--demo" in sys.argv or "--simple" in sys.argv
        if demo:
            print("GeoAgent 简单模式（离线，无需 API key），输入 help 查看指令，quit 退出。")
            while True:
                try:
                    line = input("你> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not line or line.lower() == "quit":
                    break
                print(_demo_reply(line, agent))
            return
        print(f"GeoAgent 已启动（online={agent.online}），输入 quit 退出。")
        while True:
            try:
                line = input("你> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line or line.lower() == "quit":
                break
            print(agent.chat(line))
    else:
        print(__doc__ or "用法: geoagent tools|run|chat [--demo]")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
