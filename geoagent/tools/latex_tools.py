"""LaTeX 工具：检测本机工具链、写入 .tex、编译 PDF。

编译默认优先 xelatex（中文文档需要），自动回退 pdflatex / latexmk；
依赖本机已安装 TeX 发行版（MiKTeX / TeX Live / tectonic 均可），
未安装时 latex_check 会给出安装指引而不是崩溃。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

from .base import registry

ENGINES = ["xelatex", "pdflatex", "lualatex", "latexmk", "tectonic"]

# 可直接使用的论文模板（支持中文，用 ctex 文档类，需 xelatex 编译）
TEMPLATE = r"""\documentclass[12pt]{ctexart}
\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\usepackage{geometry}
\geometry{a4paper, margin=2.5cm}

\title{%TITLE%}
\author{%AUTHOR%}
\date{\today}

\begin{document}
\maketitle

\section{引言}
在此处开始撰写你的内容……

\section{方法}
公式示例：
\begin{equation}
  v(t) = v_0 + \int_0^t a(\tau)\,d\tau
\end{equation}

\end{document}
"""


def _which(engine: str) -> str | None:
    return shutil.which(engine)


def _run(cmd: list, cwd: str, timeout: int) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 1, f"ERROR: 编译超时（>{timeout}s）"
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


@registry.register(category="latex")
def latex_check() -> str:
    """检测本机可用的 LaTeX 引擎（xelatex/pdflatex/latexmk 等）及版本。"""
    found = []
    for eng in ENGINES:
        path = _which(eng)
        if not path:
            continue
        code, out = _run([eng, "--version"], cwd=os.getcwd(), timeout=20)
        ver = out.strip().splitlines()[0] if out.strip() else ""
        found.append(f"{eng}: {path}\n    {ver[:90]}")
    if not found:
        return (
            "未检测到任何 LaTeX 引擎。安装建议：\n"
            "  Windows: winget install MiKTeX.MiKTeX\n"
            "  或轻量方案: winget install Tectonic.Tectonic\n"
            "安装后重新运行 latex_check。"
        )
    return "本机 LaTeX 工具链:\n" + "\n".join(found)


@registry.register(category="latex")
def latex_write(filename: str, title: str = "Untitled", author: str = "", content: str = "") -> str:
    """在工作目录写入一个 LaTeX 文件；content 为空时使用中文论文模板（ctexart）。"""
    if not filename.endswith(".tex"):
        filename += ".tex"
    body = content if content.strip() else TEMPLATE
    body = body.replace("%TITLE%", title).replace("%AUTHOR%", author)
    root = os.getcwd()
    full = os.path.abspath(os.path.join(root, filename))
    if os.path.commonpath([root, full]) != root:
        return f"ERROR: 路径越出工作目录: {filename}"
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(body)
    return f"已写入 {filename}（{len(body)} 字符）。使用 latex_compile 将其编译为 PDF。"


@registry.register(category="latex")
def latex_compile(filename: str, engine: str = "xelatex", passes: int = 2, timeout_sec: int = 180) -> str:
    """把工作目录内的 .tex 编译为 PDF。

    默认 xelatex 跑两遍（解决目录/交叉引用）；引文场景可改用 latexmk。
    编译失败时返回日志末尾的错误行，方便直接定位语法问题。
    """
    if not filename.endswith(".tex"):
        filename += ".tex"
    root = os.getcwd()
    full = os.path.abspath(os.path.join(root, filename))
    if not os.path.isfile(full):
        return f"ERROR: 文件不存在: {filename}（可先用 latex_write 生成）"
    with open(full, encoding="utf-8", errors="replace") as f:
        src = f.read()
    # ctex 等中文文档类只能用 xelatex 编译
    if engine != "xelatex" and re.search(r"\\documentclass(\[[^\]]*\])?\{ctex", src):
        engine = "xelatex"
    exe = _which(engine)
    if not exe:
        alts = [e for e in ENGINES if _which(e)]
        if not alts:
            return "ERROR: 本机没有可用的 LaTeX 引擎，请先运行 latex_check 查看安装指引"
        engine = alts[0]
        exe = _which(engine)
    if engine == "latexmk":
        cmd = [exe, "-interaction=nonstopmode", filename]
    elif engine == "tectonic":
        cmd = [exe, filename]
    else:
        cmd = [exe, "-interaction=nonstopmode", "-halt-on-error", filename]

    log_tail = ""
    for i in range(1 if engine in ("latexmk", "tectonic") else max(1, passes)):
        code, log_tail = _run(cmd, cwd=root, timeout=timeout_sec)
        if code != 0:
            break

    pdf = filename[:-4] + ".pdf"
    if code == 0 and os.path.isfile(os.path.join(root, pdf)):
        size = os.path.getsize(os.path.join(root, pdf))
        return f"编译成功: {pdf}（{size:,} 字节, 引擎 {engine}, {passes} 遍）"
    errors = [
        ln for ln in log_tail.splitlines()
        if ln.startswith("!") or "Error" in ln or "错误" in ln
    ][:8]
    return (
        f"ERROR: 编译失败（引擎 {engine}）。日志关键行:\n" +
        ("\n".join(errors) if errors else log_tail[-800:])
    )
