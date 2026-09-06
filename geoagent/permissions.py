"""权限管理：像 Claude Code / Codex 一样的多级工具权限模式。

四种模式（持久化在 config.json，CLI 用 /mode、GUI 用下拉切换）：
  readonly  只读 —— 仅允许检索/分析/计算类工具，禁止一切写入、执行与联网
  standard  标准 —— 工作区内读写、执行代码、联网查询全部允许（默认）
  ask       谨慎 —— 读允许；写入/执行/联网逐次弹窗确认（GUI）或终端确认（CLI）
  full      自主 —— 全部允许，并跳过一切确认（供长任务无人值守）

风险分级：network（联网）> exec（执行代码/编译）> write（写文件/建索引）> read。
"""

from __future__ import annotations

from collections.abc import Callable

READONLY_TOOLS = {
    # 文件与数据（只读）
    "list_files",
    "read_file",
    "segy_info",
    "segy_trace_amplitude",
    "segy_extract_window",
    "las_curves",
    "las_read_curve",
    "arxiv_search",
    "fetch_usgs_earthquakes",
    "fetch_iris_events",
    "citation_lookup",
    # 信号/正演/处理（纯计算）
    "signal_spectrum",
    "signal_filter",
    "velocity_to_depth",
    "tuning_thickness",
    "ricker_wavelet",
    "synthetic_trace",
    "nmo_velocity_scan",
    "nmo_correct",
    "stack_traces",
    "diffraction_stack_migrate",
    # ML（纯计算/预测）
    "dataset_profile",
    "quick_train",
    "cross_validate",
    "pca_reduce",
    "feature_correlation",
    "confusion_report",
    "mlp_predict",
    # 文献与记忆（只读检索）
    "lit_search",
    "lit_status",
    "lit_compare",
    "list_notes",
    "read_note",
    "search_notes",
    "latex_check",
}

NETWORK_TOOLS = {
    "arxiv_search",
    "arxiv_download",
    "fetch_usgs_earthquakes",
    "fetch_iris_events",
    "fetch_iris_waveform",
    "citation_lookup",
    "lit_ingest_crossref",
    "lit_ask",
    "lit_compare",
    "see_image",
    "lit_reindex",  # provider 后端需要联网嵌入
}

EXEC_TOOLS = {"run_python", "latex_compile"}


def risk_of(tool_name: str) -> str:
    """工具风险分级：network / exec / write / read。"""
    if tool_name in NETWORK_TOOLS:
        return "network"
    if tool_name in EXEC_TOOLS:
        return "exec"
    if tool_name in READONLY_TOOLS:
        return "read"
    return "write"


RISK_LABEL = {"network": "联网", "exec": "执行代码", "write": "写入", "read": "读取"}


class PermissionPolicy:
    """按模式判定工具是否可执行。ask 模式通过 confirm 回调逐次询问。"""

    def __init__(self, mode: str = "standard", confirm: Callable[[str, str], bool] | None = None) -> None:
        self.mode = mode
        self.confirm = confirm  # (工具名, 风险说明) -> bool

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def check(self, tool_name: str) -> tuple[bool, str]:
        """返回 (是否允许, 拒绝原因)。允许时 reason 为空。"""
        risk = risk_of(tool_name)
        if self.mode == "full" or self.mode == "standard":
            return True, ""
        if self.mode == "readonly":
            if risk == "read":
                return True, ""
            return False, f"只读模式禁止{RISK_LABEL[risk]}类操作"
        if self.mode == "ask":
            if risk == "read":
                return True, ""
            if self.confirm is None:
                return False, "谨慎模式需要确认，但当前界面不支持确认弹窗"
            if self.confirm(tool_name, RISK_LABEL[risk]):
                return True, ""
            return False, "你在确认弹窗中拒绝了本次调用"
        return False, f"未知权限模式 {self.mode!r}"
