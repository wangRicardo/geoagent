"""Ricardo Agent 命令行入口。

用法:
  ricardo [文件夹] chat [--demo]   # 在指定文件夹开启对话（默认当前目录）
  ricardo tools                    # 列出全部工具
  ricardo run <工具> "k=v; ..."    # 直接调用工具
"""

import json
import re
import sys

from . import GeoAgent, registry
from .config import PROVIDERS, THINKING_LEVELS

BANNER = r"""
    ██╗██╗   ██╗███████╗
    ██║██║   ██║██╔════╝      █████╗  ██████╗ ███████╗███╗   ██╗████████╗
    ██║██║   ██║███████╗    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
██╗ ██║██║   ██║╚════██║    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║
╚█████╔╝╚██████╔╝███████║    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║
 ╚════╝  ╚═════╝ ╚══════╝    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║
                             ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝
        地球物理 × 机器学习 · 你的个人研究 Agent · v0.4.0
"""

SLASH_HELP = """对话内命令:
  /help                    显示本帮助
  /status                  当前提供商/模型/思考强度/工作目录
  /provider [名称]          查看({p})或切换提供商
  /model [名称]             查看(内置+自定义)或切换模型
  /thinking [级别]          思考强度: {t}
  /cd [路径]                切换工作文件夹（类似 codex 选工作区）
  /clear                   清空对话历史
  /export [文件名.md]       导出对话为 Markdown 研究日志（存到工作区）
  /tools                   列出工具
  /config                  配置保存位置: ~/.geoagent/config.json
其余输入都会发给模型。"quit" 或 Ctrl+C 退出。""".format(
    p=", ".join(PROVIDERS), t=", ".join(THINKING_LEVELS)
)


def _handle_slash(line: str, agent: GeoAgent) -> str | None:
    """处理斜杠命令；返回要打印的文本，None 表示交给 LLM。"""
    if not line.startswith("/"):
        return None
    parts = line.split(maxsplit=1)
    cmd, arg = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")
    cfg = agent.config
    if cmd == "/help":
        return SLASH_HELP
    if cmd == "/status":
        return f"{cfg.status()}\n工作目录: {agent.workdir}\n对话轮数: {len(agent.history)}"
    if cmd == "/provider":
        return cfg.set_provider(arg) if arg else f"可用提供商: {', '.join(PROVIDERS)}\n当前: {cfg.provider}"
    if cmd == "/model":
        if not arg:
            return "可用模型: " + ", ".join(cfg.models())
        out = cfg.set_model(arg)
        return out + "\n（提示: 输入 /reload 不需要——下一条消息自动生效）"
    if cmd == "/thinking":
        return cfg.set_thinking(arg) if arg else f"思考强度当前: {cfg.thinking}，可选: {', '.join(THINKING_LEVELS)}"
    if cmd == "/cd":
        return agent.set_workdir(arg) if arg else f"工作目录: {agent.workdir}"
    if cmd == "/clear":
        agent.history = []
        return "对话历史已清空。"
    if cmd == "/export":
        path = arg or f"chat_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        return agent.export_transcript(path)
    if cmd == "/tools":
        return "\n".join(f"[{t.category}] {t.name}: {t.description[:50]}" for t in registry.list())
    if cmd == "/config":
        from .config import CONFIG_PATH
        return f"配置文件: {CONFIG_PATH}（提供商/模型/思考强度会自动保存）"
    return f"未知命令 {cmd}。输入 /help 查看命令列表。"


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
            "  看文件 / 读文件 x.py / 写文件 x.py 内容=...\n"
            "  记笔记 标题=... 内容=... [标签=a,b]\n"
            "  看笔记 [标签] / 搜笔记 关键词\n"
            "接入真实 LLM 后可完全自然语言对话。"
        )
    if "工具" in low or low == "tools":
        return "\n".join(f"[{t.category}] {t.name}: {t.description[:50]}" for t in registry.list())
    if low.startswith(("看文件", "ls", "list_files")):
        return _run("list_files", {"subdir": _kv(line, "路径") or "."})
    if low.startswith(("读文件", "read_file")):
        p = _kv(line, "路径") or low.replace("读文件", "").replace("read_file", "").strip()
        return _run("read_file", {"path": p}) if p else "请给出路径，例如：读文件 data.py"
    if low.startswith(("写文件", "write_file")):
        p, c = _kv(line, "路径"), _kv(line, "内容")
        return _run("write_file", {"path": p, "content": c}) if p and c else "用法：写文件 路径=x.py 内容=print(1)"
    if low.startswith(("记笔记", "save_note")):
        return _run("save_note", {
            "title": _kv(line, "标题") or "未命名笔记",
            "content": _kv(line, "内容") or "",
            "tags": _kv(line, "标签") or "",
        })
    if low.startswith(("看笔记", "list_notes")):
        return _run("list_notes", {"tag": _kv(line, "标签") or ""})
    if low.startswith(("搜笔记", "search_notes")):
        kw = low.replace("搜笔记", "").replace("search_notes", "").strip()
        return _run("search_notes", {"keyword": kw}) if kw else "请给出关键词，例如：搜笔记 调谐"
    if re.search(r"ricker|子波", low):
        return _run("ricker_wavelet", {"freq_hz": _num(line, "freq_hz", "主频", default=30.0)})
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
    return (
        "（简单模式）我没听懂这句。输入 help 查看支持的指令；"
        "或配置 API 密钥使用完整 LLM 对话。"
    )


def _num(line: str, eng: str, cn: str, default: float) -> float:
    m = re.search(rf"{eng}\s*=\s*([\d.]+)", line.lower()) or re.search(rf"{cn}\s*=\s*([\d.]+)", line)
    return float(m.group(1)) if m else default


def _kv(line: str, key: str) -> str:
    m = re.search(rf"{key}\s*=\s*(\S+)", line)
    return m.group(1) if m else ""


def main() -> None:
    args = [a for a in sys.argv[1:] if a not in ("--demo", "--simple")]
    demo = len(args) != len(sys.argv[1:])
    workdir = next((a for a in args if not a.startswith("-") and
                    a not in ("tools", "run", "chat", "gui", "help")), None)
    cmd = next((a for a in args if a in ("tools", "run", "chat", "gui")), "chat" if workdir else "tools")
    if cmd == "gui":
        from .gui import launch
        launch(workdir)
        return
    agent = GeoAgent(workdir=workdir)

    if cmd == "tools":
        print(BANNER)
        for t in registry.list():
            print(f"[{t.category:10s}] {t.name}: {t.description[:60]}")
    elif cmd == "run":
        rest = args[args.index("run") + 1:]
        name = rest[0]
        kv_args: dict = {}
        if len(rest) > 1:
            for pair in rest[1].split(";"):
                pair = pair.strip()
                if not pair:
                    continue
                k, v = pair.split("=", 1)
                try:
                    v = json.loads(v)
                except json.JSONDecodeError:
                    pass
                kv_args[k] = v
        print(agent.run_tool(name, kv_args))
    elif cmd == "chat":
        print(BANNER)
        print(f"  提供商: {agent.config.provider} | 模型: {agent.config.model} | "
              f"思考强度: {agent.config.thinking} | online={agent.online}")
        print(f"  工作目录: {agent.workdir}")
        print("  输入 /help 查看对话内命令，quit 退出。\n")
        while True:
            try:
                line = input("你> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line or line.lower() == "quit":
                break
            if line.startswith("/"):
                print(_handle_slash(line, agent))
                continue
            if demo:
                print(_demo_reply(line, agent))
                continue
            try:
                print(agent.chat(line))
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR: {exc}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
