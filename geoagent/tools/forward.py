"""地震正演模拟工具。

合成记录是连接"地质认识"与"地震响应"的桥梁，也是 ML 训练数据的重要来源。
"""

from __future__ import annotations

from typing import List

import numpy as np

from .base import registry


@registry.register(category="geophysics")
def ricker_wavelet(
    freq_hz: float = 30.0,
    dt_ms: float = 1.0,
    length_ms: float = 100.0,
) -> str:
    """生成 Ricker 子波（地震解释中最常用的零相位子波）。"""
    if freq_hz <= 0:
        return "ERROR: freq_hz 必须为正"
    t = np.arange(-length_ms / 2, length_ms / 2 + dt_ms / 2, dt_ms) / 1000.0
    a = np.pi ** 2 * freq_hz ** 2
    w = (1 - 2 * a * t ** 2) * np.exp(-a * t ** 2)
    peak_t = t[int(np.argmax(np.abs(w)))] * 1000
    return (
        f"Ricker 子波: 主频 {freq_hz} Hz, dt={dt_ms} ms, {t.size} 个采样\n"
        f"振幅峰值位于 t={peak_t:.1f} ms（应为 0 ms 附近，零相位）\n"
        f"采样值: {np.round(w, 4).tolist()}"
    )


@registry.register(category="geophysics")
def synthetic_trace(
    impedance: List[float],
    wavelet_freq_hz: float = 30.0,
    dt_ms: float = 1.0,
) -> str:
    """由一维波阻抗序列生成合成地震道：反射系数与 Ricker 子波褶积。

    impedance 按时间采样序给出各界面波阻抗（单位任意，只影响反射系数符号）。
    """
    if len(impedance) < 3:
        return "ERROR: impedance 至少需要 3 个采样点"
    z = np.asarray(impedance, dtype=float)
    rc = np.zeros_like(z)
    rc[1:] = (z[1:] - z[:-1]) / (z[1:] + z[:-1])
    n = int(100.0 / dt_ms)
    t = np.arange(-n / 2, n / 2) / 1000.0
    a = np.pi ** 2 * wavelet_freq_hz ** 2
    w = (1 - 2 * a * t ** 2) * np.exp(-a * t ** 2)
    trace = np.convolve(rc, w, mode="same")
    return (
        f"合成地震道: {z.size} 层, 主频 {wavelet_freq_hz} Hz\n"
        f"反射系数: {np.round(rc, 4).tolist()}\n"
        f"合成道: {np.round(trace, 4).tolist()}"
    )


@registry.register(category="geophysics")
def tuning_thickness(freq_hz: float, vp_ms: float = 3000.0) -> str:
    """计算调谐厚度：分辨率极限 ≈ 波长/4，返回可分辨的最小地层厚度。"""
    if freq_hz <= 0 or vp_ms <= 0:
        return "ERROR: freq_hz 与 vp_ms 必须为正"
    wavelength = vp_ms / freq_hz
    return (
        f"主频 {freq_hz} Hz @ vp={vp_ms} m/s: 波长 {wavelength:.1f} m\n"
        f"垂向分辨率极限(λ/4): {wavelength / 4:.1f} m"
    )
