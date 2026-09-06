"""可观测性：统一日志（滚动文件）与 GUI 崩溃捕获。

日志位置：~/.geoagent/logs/ricardo.log（5MB x 3 个轮转）。
工业级要求：任何未捕获异常都必须落盘可追溯，而不是无声消失。
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path.home() / ".geoagent" / "logs"
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

_configured = False


def setup_logging(level: int = logging.INFO) -> Path:
    """初始化滚动日志；重复调用安全。返回日志文件路径。"""
    global _configured
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / "ricardo.log"
    if not _configured:
        handler = RotatingFileHandler(path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter(_FORMAT))
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(level)
        _configured = True
    return path


def log_uncaught(exc_type, exc_value, exc_tb) -> None:
    """sys.excepthook：未捕获异常写日志并继续交还默认处理。"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    logging.getLogger("ricardo.crash").critical("未捕获异常", exc_info=(exc_type, exc_value, exc_tb))
    sys.__excepthook__(exc_type, exc_value, exc_tb)


def gui_crash_hook(exc_type, exc_value, exc_tb) -> None:
    """GUI 专用钩子：写日志 + 弹窗告知用户日志位置，程序不闪退。"""
    log_uncaught(exc_type, exc_value, exc_tb)
    if exc_type is KeyboardInterrupt:
        return
    try:
        import traceback
        from tkinter import messagebox

        traceback.print_exception(exc_type, exc_value, exc_tb)
        messagebox.showerror(
            "Ricardo Agent 遇到错误",
            f"{exc_type.__name__}: {exc_value}\n\n详细信息已写入日志:\n{LOG_DIR / 'ricardo.log'}",
        )
    except Exception:  # noqa: BLE001 - 弹窗失败时保证日志已落盘
        pass
