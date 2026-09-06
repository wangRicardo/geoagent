"""地球物理领域工具。

SEG-Y / LAS 相关工具依赖 segyio / lasio，未安装时调用会返回提示信息
而不是让整个 agent 崩溃；numpy / scipy 为主的通用信号处理工具开箱即用。
"""

from __future__ import annotations

import os

import numpy as np

from .base import registry


def _missing(lib: str, pip_name: str = "") -> str:
    pip_name = pip_name or lib
    return f"ERROR: 需要 {lib} 库。请先安装: pip install {pip_name}"


# ---------------------------------------------------------------------------
# SEG-Y 地震数据
# ---------------------------------------------------------------------------


@registry.register(category="geophysics")
def segy_info(path: str) -> str:
    """读取 SEG-Y 文件的基本信息：inline/xline 范围、采样率、道数等。"""
    try:
        import segyio
    except ImportError:
        return _missing("segyio")
    with segyio.open(path, ignore_geometry=True) as f:
        header = f.bin[segyio.BinField.Interval]  # 微秒
        return (
            f"文件: {path}\n"
            f"道数: {f.tracecount}\n"
            f"每道采样点: {f.samples.size}\n"
            f"采样间隔: {header} us ({header / 1000:.1f} ms)\n"
            f"时间范围: {f.samples[0]:.1f} ~ {f.samples[-1]:.1f} ms"
        )


@registry.register(category="geophysics")
def segy_trace_amplitude(path: str, trace_index: int) -> str:
    """提取 SEG-Y 文件中指定道的振幅序列（前 50 个采样 + 统计量）。"""
    try:
        import segyio
    except ImportError:
        return _missing("segyio")
    with segyio.open(path, ignore_geometry=True) as f:
        if not (0 <= trace_index < f.tracecount):
            return f"ERROR: trace_index 超出范围 [0, {f.tracecount - 1}]"
        amp = f.trace[trace_index]
    return (
        f"道 {trace_index}: n={amp.size}, "
        f"min={amp.min():.4g}, max={amp.max():.4g}, "
        f"mean={amp.mean():.4g}, rms={np.sqrt(np.mean(amp**2)):.4g}\n"
        f"前50采样: {np.round(amp[:50], 4).tolist()}"
    )


@registry.register(category="geophysics")
def segy_write(
    traces: list[list[float]],
    dt_ms: float = 1.0,
    filename: str = "output.sgy",
) -> str:
    """把多道地震数据写入 SEG-Y 文件（IEEE float32，rev1）。

    traces 每行一道。适合保存合成记录、滤波/处理后的道集，
    写出的文件可被 segyio、Petrel 等标准软件读取。
    """
    try:
        import segyio
    except ImportError:
        return _missing("segyio")
    data = np.asarray(traces, dtype=np.float32)
    if data.ndim != 2:
        return f"ERROR: traces 需是二维列表（每行一道），当前 {data.ndim} 维"
    root = os.getcwd()
    full = os.path.abspath(os.path.join(root, filename))
    if os.path.commonpath([root, full]) != root:
        return f"ERROR: 路径越出工作目录: {filename}"
    nt, ns = data.shape
    spec = segyio.spec()
    spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING
    spec.format = 5  # IEEE float32
    spec.samples = np.arange(ns) * dt_ms  # segyio.create 按毫秒解释，写为 µs 间隔
    spec.tracecount = nt
    spec.ilines = np.arange(nt) + 1  # 每道视为一条 inline，方便几何读取
    with segyio.create(full, spec) as f:
        for i in range(nt):
            f.trace[i] = data[i]
            f.header[i] = {
                segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,
                segyio.TraceField.offset: 0,
                segyio.TraceField.SourceX: i * 1000,  # 假道距 1 km，方便几何浏览
                segyio.TraceField.INLINE_3D: i + 1,
            }
    return (
        f"SEG-Y 已写入: {filename}（{os.path.getsize(full):,} 字节）\n"
        f"  {nt} 道 x {ns} 采样, dt={dt_ms}ms, format=IEEE float32"
    )


@registry.register(category="geophysics")
def segy_extract_window(path: str, trace_index: int, t_start_ms: float, t_end_ms: float) -> str:
    """提取 SEG-Y 指定道在时间窗 [t_start_ms, t_end_ms] 内的振幅与统计量。"""
    try:
        import segyio
    except ImportError:
        return _missing("segyio")
    with segyio.open(path, ignore_geometry=True) as f:
        if not (0 <= trace_index < f.tracecount):
            return f"ERROR: trace_index 超出范围 [0, {f.tracecount - 1}]"
        t = f.samples
        if t_start_ms < t[0] or t_end_ms > t[-1] or t_start_ms >= t_end_ms:
            return f"ERROR: 时间窗无效（文件范围 {t[0]:.0f}~{t[-1]:.0f} ms）"
        amp = f.trace[trace_index]
        w = amp[(t >= t_start_ms) & (t <= t_end_ms)]
    return (
        f"道 {trace_index} @ {t_start_ms:.0f}~{t_end_ms:.0f} ms: {w.size} 个采样\n"
        f"min={w.min():.4g}, max={w.max():.4g}, rms={np.sqrt(np.mean(w**2)):.4g}\n"
        f"数值: {np.round(w[:100], 4).tolist()}"
    )


# ---------------------------------------------------------------------------
# LAS 测井数据
# ---------------------------------------------------------------------------


@registry.register(category="geophysics")
def las_curves(path: str) -> str:
    """列出 LAS 测井文件中所有曲线名、单位与深度范围。"""
    try:
        import lasio
    except ImportError:
        return _missing("lasio")
    las = lasio.read(path)
    lines = [f"井: {las.well.WELL.value if 'WELL' in las.well else '?'}"]
    for c in las.curves:
        lines.append(f"  {c.mnemonic:12s} unit={c.unit!r:8s} 有效样本={np.isfinite(las[c.mnemonic]).sum()}")
    d = las[0]
    finite = d[np.isfinite(d)]
    if finite.size:
        lines.append(f"深度范围: {finite.min():.2f} ~ {finite.max():.2f} {las.curves[0].unit or ''}")
    return "\n".join(lines)


@registry.register(category="geophysics")
def las_read_curve(path: str, curve: str, max_points: int = 200) -> str:
    """读取 LAS 文件中某条测井曲线的深度-数值对（截断到 max_points）。"""
    try:
        import lasio
    except ImportError:
        return _missing("lasio")
    las = lasio.read(path)
    if curve not in [c.mnemonic for c in las.curves]:
        return f"ERROR: 曲线 {curve!r} 不存在。可用: {[c.mnemonic for c in las.curves]}"
    depth, value = las[0], las[curve]
    mask = np.isfinite(value)
    depth, value = depth[mask], value[mask]
    if depth.size > max_points:
        idx = np.linspace(0, depth.size - 1, max_points).astype(int)
        depth, value = depth[idx], value[idx]
    return (
        f"曲线 {curve}: {depth.size} 个有效点, "
        f"范围 {np.min(value):.4g} ~ {np.max(value):.4g}\n"
        f"深度[{las.curves[0].unit or ''}]: {np.round(depth, 2).tolist()}\n"
        f"数值: {np.round(value, 4).tolist()}"
    )


# ---------------------------------------------------------------------------
# 通用信号处理（地震道 / 时序都适用）
# ---------------------------------------------------------------------------


@registry.register(category="geophysics")
def signal_spectrum(data: list[float], dt_ms: float = 1.0) -> str:
    """计算实数序列的幅值谱，返回主频及峰值频率列表。"""
    x = np.asarray(data, dtype=float)
    n = x.size
    amp = np.abs(np.fft.rfft(x - x.mean())) / n
    freq = np.fft.rfftfreq(n, d=dt_ms / 1000.0)
    order = np.argsort(amp)[::-1][:5]
    peaks = [
        {"freq_hz": round(float(freq[i]), 2), "amp": round(float(amp[i]), 4)} for i in order if freq[i] > 0
    ]
    return f"主频峰值(Hz, 幅值): {peaks}"


@registry.register(category="geophysics")
def signal_filter(
    data: list[float],
    dt_ms: float = 1.0,
    lowcut_hz: float = 0.0,
    highcut_hz: float = 0.0,
    order: int = 4,
) -> str:
    """Butterworth 带通滤波（lowcut/highcut 任一为 0 表示该侧不设限）。"""
    try:
        from scipy.signal import butter, filtfilt
    except ImportError:
        return _missing("scipy")
    x = np.asarray(data, dtype=float)
    fs = 1000.0 / dt_ms
    nyq = fs / 2.0
    lo, hi = lowcut_hz / nyq, highcut_hz / nyq
    if lo > 0 and hi > 0:
        b, a = butter(order, [lo, hi], btype="band")
    elif hi > 0:
        b, a = butter(order, hi, btype="low")
    elif lo > 0:
        b, a = butter(order, lo, btype="high")
    else:
        return "ERROR: lowcut_hz / highcut_hz 至少设置一个"
    y = filtfilt(b, a, x)
    return f"滤波完成: 输入rms={np.sqrt(np.mean(x**2)):.4g}, 输出rms={np.sqrt(np.mean(y**2)):.4g}\n结果: {np.round(y, 4).tolist()}"


@registry.register(category="geophysics")
def velocity_to_depth(vp_ms: float, two_way_time_ms: list[float]) -> str:
    """用恒定速度把双程旅行时 (TWT) 转换为深度，返回每层的深度列表。"""
    depth = [round(v * t / 2000.0, 2) for v, t in [(vp_ms, t) for t in two_way_time_ms]]
    return f"vp={vp_ms} m/s, TWT={two_way_time_ms} ms -> 深度(m): {depth}"
