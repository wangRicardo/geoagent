"""GeoAgent 命令行入口（作为包模块，支持 pip 安装后的 `geoagent` 命令）。"""

import json
import sys

from . import GeoAgent, registry


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
        print(__doc__ or "用法: geoagent tools|run|chat")


if __name__ == "__main__":
    main()
