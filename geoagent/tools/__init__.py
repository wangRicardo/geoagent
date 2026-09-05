"""导入即注册所有内置工具。"""

from . import citation_tools, data_tools, forward, fs_tools, geophysics, latex_tools, ml_tools, notes, plot_tools  # noqa: F401
from .base import ToolRegistry, registry  # noqa: F401
