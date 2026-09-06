"""图片生成工具：地震剖面、测井曲线、交会图与时序图。

统一约定：函数接收数值列表（LLM 从上游工具拿到的数据可直接传入），
保存 PNG 到工作目录并返回文件路径与数据摘要，方便 agent 串联流程、
也让用户在 /cd 的文件夹里直接看到产物。
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from .base import registry


def _save(fig, filename: str) -> str:
    root = os.getcwd()
    if not filename.endswith(".png"):
        filename += ".png"
    full = os.path.abspath(os.path.join(root, filename))
    if os.path.commonpath([root, full]) != root:
        raise ValueError(f"路径越出工作目录: {filename}")
    fig.savefig(full, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return full


def _auto_vision_check(path: str) -> str:
    """出图后自动做一次视觉质检（需要 API 密钥）；失败静默跳过不阻塞出图。

    设环境变量 RICARDO_AUTO_VISION=0 可关闭。
    """
    if os.environ.get("RICARDO_AUTO_VISION", "1") == "0":
        return ""
    try:
        from .vision_tools import see_image
        out = see_image(path, "快速质检：这张图是否达到论文插图水准？只给一句话结论。")
        return "\n  👁 视觉自查: " + out.splitlines()[0] if out else ""
    except Exception:  # noqa: BLE001
        return ""


@registry.register(category="plot")
def plot_seismic_section(
    traces: List[List[float]],
    dt_ms: float = 1.0,
    filename: str = "seismic_section.png",
    title: str = "Seismic Section",
) -> str:
    """绘制二维地震剖面（每行一道，波形填充图），保存 PNG 并返回路径。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.asarray(traces, dtype=float)
    if data.ndim != 2:
        return f"ERROR: traces 需是二维列表（每行一道），当前 {data.ndim} 维"
    nt, ns = data.shape
    t = np.arange(ns) * dt_ms
    norm = np.abs(data).max() or 1.0
    fig, ax = plt.subplots(figsize=(8, 6))
    # 变面积填充：正振幅黑、负振幅红（SEG 惯例的简化版）
    clip = 1.5
    for i in range(nt):
        tr = data[i] / norm
        ax.fill_betweenx(t, i + tr, i, where=tr > 0, color="k", lw=0)
        ax.fill_betweenx(t, i + tr, i, where=tr < 0, color="firebrick", lw=0)
    ax.set_xlim(-0.5, nt - 0.5)
    ax.set_ylim(t[-1], t[0])
    ax.set_xlabel("Trace")
    ax.set_ylabel("Time (ms)")
    ax.set_title(title)
    path = _save(fig, filename)
    plt.close(fig)
    return f"剖面图已保存: {path}\n  {nt} 道 x {ns} 采样, dt={dt_ms}ms, 最大绝对振幅={norm:.4g}"


@registry.register(category="plot")
def plot_crossplot(
    x: List[float],
    y: List[float],
    x_label: str = "X",
    y_label: str = "Y",
    color_by: Optional[List[float]] = None,
    filename: str = "crossplot.png",
) -> str:
    """绘制交会图（散点图），可用第三变量着色；自动计算 Pearson 相关系数。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xv, yv = np.asarray(x, float), np.asarray(y, float)
    if xv.size != yv.size or xv.size < 2:
        return "ERROR: x 与 y 长度需一致且至少 2 个点"
    fig, ax = plt.subplots(figsize=(7, 6))
    if color_by is not None:
        cv = np.asarray(color_by, float)
        sc = ax.scatter(xv, yv, c=cv, cmap="viridis", s=18, alpha=.85)
        fig.colorbar(sc, ax=ax, label="color")
    else:
        ax.scatter(xv, yv, s=18, alpha=.75, c="#22d3ee")
    r = np.corrcoef(xv, yv)[0, 1]
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} vs {x_label}  (r = {r:.3f})")
    ax.grid(alpha=.25)
    path = _save(fig, filename)
    plt.close(fig)
    return f"交会图已保存: {path}\n  {xv.size} 点, Pearson r = {r:.4f}"


@registry.register(category="plot")
def plot_well_logs(
    curves: List[List[float]],
    names: List[str],
    depth_unit: str = "m",
    filename: str = "well_logs.png",
) -> str:
    """并排绘制多条测井曲线（共享深度轴，深度向下增加）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(curves)
    if n == 0 or len(names) != n:
        return "ERROR: curves 与 names 数量需一致且非空"
    depth = np.arange(len(curves[0]), dtype=float)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 7), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, cur, name in zip(axes, curves, names):
        c = np.asarray(cur, float)
        ax.plot(c, depth, lw=1.1, color="#818cf8")
        ax.set_xlabel(name)
        ax.grid(alpha=.25)
    axes[0].set_ylabel(f"Depth ({depth_unit})")
    axes[0].invert_yaxis()
    fig.suptitle("Well Log Curves")
    path = _save(fig, filename)
    plt.close(fig)
    return f"测井曲线图已保存: {path}\n  {n} 条曲线 x {depth.size} 个深度点"


@registry.register(category="plot")
def plot_time_series(
    series: List[List[float]],
    labels: List[str],
    dt_s: float = 1.0,
    filename: str = "timeseries.png",
    title: str = "Time Series",
) -> str:
    """叠加绘制多条时间序列（波形对比、滤波前后对比等场景）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(series) != len(labels) or not series:
        return "ERROR: series 与 labels 数量需一致且非空"
    t = np.arange(len(series[0])) * dt_s
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for s, lab in zip(series, labels):
        ax.plot(t, np.asarray(s, float), lw=1.0, label=lab)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=.25)
    path = _save(fig, filename)
    plt.close(fig)
    return f"时序图已保存: {path}\n  {len(series)} 条曲线 x {t.size} 点, dt={dt_s}s"


@registry.register(category="plot")
def plot_spectrogram(
    data: List[float],
    dt_ms: float = 1.0,
    filename: str = "spectrogram.png",
    title: str = "Time-Frequency Spectrogram",
) -> str:
    """时频谱图（STFT），展示信号频率成分随时间的变化（地震道/测井曲线均适用）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    x = np.asarray(data, dtype=float)
    fs = 1000.0 / dt_ms
    f, t, Sxx = spectrogram(x, fs=fs, nperseg=min(128, x.size))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="auto", cmap="inferno")
    fig.colorbar(im, ax=ax, label="Power (dB)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    path = _save(fig, filename)
    plt.close(fig)
    peak_f = float(f[np.argmax(Sxx.mean(axis=1))])
    return (f"时频谱图已保存: {path}\n  {x.size} 采样, 频带 0~{fs/2:.0f} Hz, "
            f"平均能量主频 {peak_f:.1f} Hz" + _auto_vision_check(path))


@registry.register(category="plot")
def plot_amplitude_section(
    traces: List[List[float]],
    dt_ms: float = 1.0,
    filename: str = "amp_section.png",
    title: str = "Amplitude Section",
    cmap: str = "seismic",
) -> str:
    """振幅彩色剖面（imshow）：波形细节的直观显示，适合对比处理前后。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.asarray(traces, dtype=float)
    if data.ndim != 2:
        return f"ERROR: traces 需是二维列表（每行一道），当前 {data.ndim} 维"
    vmax = float(np.abs(data).max()) or 1.0
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(data.T, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                   extent=[0, data.shape[0], data.shape[1] * dt_ms, 0])
    fig.colorbar(im, ax=ax, label="Amplitude")
    ax.set_xlabel("Trace")
    ax.set_ylabel("Time (ms)")
    ax.set_title(title)
    path = _save(fig, filename)
    plt.close(fig)
    return (f"振幅剖面已保存: {path}\n  {data.shape[0]} 道 x {data.shape[1]} 采样, "
            f"色标 ±{vmax:.3g}" + _auto_vision_check(path))


@registry.register(category="plot")
def plot_three_component(
    data_e: List[float],
    data_n: List[float],
    data_z: List[float],
    dt_s: float = 0.01,
    filename: str = "three_component.png",
    station: str = "Station",
) -> str:
    """三分量地震图（E/N/Z 并排），台站事件分析的标准展示。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comps = [np.asarray(d, dtype=float) for d in (data_e, data_n, data_z)]
    if len({c.size for c in comps}) != 1:
        return "ERROR: 三分量长度必须一致"
    t = np.arange(comps[0].size) * dt_s
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    for ax, c, name in zip(axes, comps, ["E", "N", "Z"]):
        ax.plot(t, c, lw=0.8, color="#22d3ee" if name != "Z" else "#f472b6")
        ax.set_ylabel(f"{name}\n(counts)")
        ax.grid(alpha=.25)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Three-Component Seismogram — {station}")
    path = _save(fig, filename)
    plt.close(fig)
    peak_z = float(np.abs(comps[2]).max())
    return (f"三分量图已保存: {path}\n  {comps[0].size} 采样, Z 分量最大振幅 {peak_z:.3g}"
            + _auto_vision_check(path))
