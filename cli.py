"""兼容入口：项目根目录运行 `python cli.py` 等价于 `geoagent` 命令。"""

from geoagent.cli import main

if __name__ == "__main__":
    main()
