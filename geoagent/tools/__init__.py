"""导入即注册所有内置工具。"""

from . import (  # noqa: F401
    citation_tools,
    data_tools,
    forward,
    fs_tools,
    geophysics,
    latex_tools,
    lit_tools,
    ml_deep,
    ml_tools,
    notes,
    plot_tools,
    processing,
    vision_tools,
)
from .base import ToolRegistry, registry  # noqa: F401
