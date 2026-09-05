"""导入即注册所有内置工具。"""

from . import forward, fs_tools, geophysics, ml_tools, notes  # noqa: F401
from .base import ToolRegistry, registry  # noqa: F401
