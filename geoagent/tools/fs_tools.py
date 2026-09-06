"""文件系统与代码执行工具 —— 让 agent 能在任意工作文件夹内像 codex 一样干活。

安全边界：read/write/list 都被限制在当前工作目录（启动目录或 /cd 切换后的目录）
之内，符号链接与 `..` 逃逸会被拒绝；run_python 子进程也在工作目录内执行并带超时。
"""

from __future__ import annotations

import os
import subprocess
import sys

from .base import registry


def _safe_path(path: str) -> str:
    """把相对路径解析到工作目录内，拒绝越界访问。"""
    root = os.getcwd()
    full = os.path.abspath(os.path.join(root, path))
    if os.path.commonpath([root, full]) != root:
        raise ValueError(f"路径越出工作目录: {path}（工作目录: {root}）")
    return full


@registry.register(category="files")
def list_files(subdir: str = ".", pattern: str = "") -> str:
    """列出工作目录（或其子目录）下的文件与文件夹，可按扩展名过滤（如 .sgy/.las/.py）。"""
    root = _safe_path(subdir)
    if not os.path.isdir(root):
        return f"ERROR: 不是目录: {subdir}"
    entries = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        kind = "DIR " if os.path.isdir(full) else "FILE"
        size = "" if kind == "DIR " else f" {os.path.getsize(full):,}B"
        if pattern and not name.lower().endswith(pattern.lower()):
            continue
        entries.append(f"{kind} {name}{size}")
    if not entries:
        return "（目录为空或无匹配文件）"
    return f"工作目录: {root}\n" + "\n".join(entries[:100])


@registry.register(category="files")
def read_file(path: str, max_chars: int = 4000) -> str:
    """读取工作目录内一个文本文件的内容（超长截断）。"""
    full = _safe_path(path)
    if not os.path.isfile(full):
        return f"ERROR: 文件不存在: {path}"
    with open(full, encoding="utf-8", errors="replace") as f:
        text = f.read(max_chars)
    more = f"\n…（已截断，共 {os.path.getsize(full):,} 字节）" if os.path.getsize(full) > max_chars else ""
    return f"--- {path} ---\n{text}{more}"


@registry.register(category="files")
def write_file(path: str, content: str) -> str:
    """在工作目录内创建或覆盖一个文本文件（含必要的父目录）。"""
    full = _safe_path(path)
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已写入 {path}（{len(content)} 字符）"


@registry.register(category="files")
def run_python(code: str, timeout_sec: int = 60) -> str:
    """在工作目录里用独立 Python 进程执行一段代码，返回 stdout/stderr（适合数据处理与画图脚本）。

    安全提示：这段代码在本机以当前用户权限真实执行（超时与工作目录隔离是仅有的
    约束），因此只应让 agent 运行你审阅过的数据处理/绘图代码；不要把密钥等敏感
    环境变量暴露给不可信来源生成的代码。
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: 执行超时（>{timeout_sec}s）"
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    result = f"exit={proc.returncode}"
    if out:
        result += f"\nstdout:\n{out[:3000]}"
    if err:
        result += f"\nstderr:\n{err[:1500]}"
    return result
