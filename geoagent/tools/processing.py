"""地震处理链工具：速度分析 → 动校正 → 叠加 → 偏移。

输入输出约定：traces 为二维列表（每行一道），时间单位 ms，速度 m/s，
偏移距 m。算法为教科书经典实现（semblance 速度谱、双曲线动校正、
均值叠加、绕射叠加偏移——零炮检距 exploding-reflector 约定），
适合教学、快速 QC 与合成数据实验；生产级处理请用专业软件。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import registry


def _as_traces(traces: List[List[float]]) -> np.ndarray:
    data = np.asarray(traces, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"traces 需是二维列表（每行一道），当前 {data.ndim} 维")
    return data


def _velocity_at(t_ms: np.ndarray, velocity) -> np.ndarray:
    """velocity 为标量或 [(t_ms, v), ...] 折线，返回逐采样速度。"""
    if np.isscalar(velocity):
        return np.full_like(t_ms, float(velocity))
    pairs = np.asarray(velocity, dtype=float)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("velocity 需为标量或 [[t_ms, v], ...] 列表")
    return np.interp(t_ms, pairs[:, 0], pairs[:, 1])


@registry.register(category="processing")
def nmo_velocity_scan(
    traces: List[List[float]],
    dt_ms: float = 2.0,
    offsets_m: Optional[List[float]] = None,
    v_min: float = 1500.0,
    v_max: float = 4500.0,
    n_velocities: int = 60,
    window_ms: float = 20.0,
    n_picks: int = 6,
) -> str:
    """CMP 道集速度扫描：计算相似系数（semblance）速度谱并给出主要速度拾取。

    offsets_m 缺省按 n 道 25~2500m 均匀生成。返回各时间的主要速度（能量聚焦处），
    可直接作为 nmo_correct 的 velocity 参数。
    """
    data = _as_traces(traces)
    nt_tr, ns = data.shape
    if offsets_m is None:
        offsets_m = np.linspace(100, 2500, nt_tr).tolist()
    x = np.asarray(offsets_m, dtype=float)
    if len(x) != nt_tr:
        return f"ERROR: offsets_m 数量 ({len(x)}) 与道数 ({nt_tr}) 不一致"
    t = np.arange(ns) * dt_ms
    half = max(1, int(window_ms / dt_ms / 2))
    velocities = np.linspace(v_min, v_max, max(10, n_velocities))

    # 相似系数：沿 NMO 双曲线窗口内能量的归一化聚焦度
    sem = np.zeros((ns, len(velocities)))
    win_offsets = np.arange(-half, half + 1) * dt_ms
    for j, v in enumerate(velocities):
        for i in range(half, ns - half, 2):  # 每 2 采样算一次
            t0 = t[i]
            tx = np.sqrt(t0 ** 2 + (x / v * 1000.0) ** 2)
            if tx.min() < half * dt_ms or tx.max() + half * dt_ms > t[-1]:
                continue
            # 各道沿其双曲线取窗口样本
            win = np.array([np.interp(tx[k] + win_offsets, t, data[k])
                            for k in range(nt_tr)])
            num = win.sum(axis=0) ** 2
            den = (win ** 2).sum(axis=0) * nt_tr + 1e-12
            sem[i, j] = num.sum() / den.sum()

    picks = []
    sem_s = sem.copy()
    for _ in range(max(1, min(n_picks, 8))):
        i, j = np.unravel_index(np.argmax(sem_s), sem_s.shape)
        if sem_s[i, j] <= 0.05:
            break
        picks.append((round(float(t[i]), 1), round(float(velocities[j])), round(float(sem_s[i, j]), 3)))
        # 抑制已拾取点附近，避免重复
        sem_s[max(0, i - int(100 / dt_ms)):i + int(100 / dt_ms), :] = 0
    if not picks:
        return "速度扫描未发现明显能量聚焦，检查数据或放宽速度范围。"
    return "速度谱拾取（t0_ms, v_m/s, semblance）:\n" + "\n".join(f"  {p}" for p in picks)


@registry.register(category="processing")
def nmo_correct(
    traces: List[List[float]],
    dt_ms: float = 2.0,
    offsets_m: Optional[List[float]] = None,
    velocity=3000.0,
    stretch_limit: float = 0.5,
) -> str:
    """双曲线动校正（NMO）：把 CMP 道集拉平到零偏移距双程时。

    velocity 为标量或 [[t_ms, v], ...]（速度拾取折线，自动插值）；
    超过 stretch_limit 的浅层远偏移距样点置零（抑制拉伸畸变）。
    """
    data = _as_traces(traces)
    nt_tr, ns = data.shape
    if offsets_m is None:
        offsets_m = np.linspace(100, 2500, nt_tr).tolist()
    x = np.asarray(offsets_m, dtype=float)
    t = np.arange(ns) * dt_ms
    out = np.zeros_like(data)
    for k in range(nt_tr):
        tx = np.sqrt(t ** 2 + (x[k] / _velocity_at(t, velocity) * 1000.0) ** 2)
        out[k] = np.interp(t, tx, data[k], left=0, right=0)
        # 拉伸切除：dt/dt0 过大的区域置零
        stretch = np.gradient(tx, t, edge_order=1)
        out[k, (stretch > 1 + stretch_limit) | ~np.isfinite(stretch)] = 0.0
    from .base import registry as _reg  # 复用结果序列化
    return (
        f"NMO 校正完成: {nt_tr} 道 x {ns} 采样, v={velocity if np.isscalar(velocity) else '折线'}\n"
        f"校正后道集(前3道各前20采样): {np.round(out[:3, :20], 4).tolist()}"
    )


@registry.register(category="processing")
def stack_traces(traces: List[List[float]]) -> str:
    """道叠加（均值），返回叠加道及其统计——NMO 后叠加可压制随机噪声 √N 倍。"""
    data = _as_traces(traces)
    stacked = data.mean(axis=0)
    rms_before = float(np.sqrt((data ** 2).mean()))
    rms_stack = float(np.sqrt((stacked ** 2).mean()))
    return (
        f"叠加完成: {data.shape[0]} 道 → 1 道 ({data.shape[1]} 采样)\n"
        f"输入道集 rms={rms_before:.4g}, 叠加道 rms={rms_stack:.4g}, "
        f"理论噪声压制 √{data.shape[0]}≈{np.sqrt(data.shape[0]):.1f}x\n"
        f"叠加道: {np.round(stacked, 4).tolist()}"
    )


@registry.register(category="processing")
def diffraction_stack_migrate(
    traces: List[List[float]],
    dt_ms: float = 2.0,
    dx_m: float = 25.0,
    v_mig: float = 3000.0,
    aperture_m: float = 2000.0,
) -> str:
    """绕射叠加偏移（零炮检距，exploding-reflector: t²=t0²+(Δx/v)²）。

    输入自激自收剖面，输出偏移剖面（同尺寸）。把绕射双曲线能量归位到
    真实反射点位置，断层/绕射点成像更清晰。
    """
    data = _as_traces(traces)
    nt_tr, ns = data.shape
    t = np.arange(ns) * dt_ms
    x = np.arange(nt_tr) * dx_m
    out = np.zeros_like(data)
    max_shift = aperture_m / v_mig * 1000.0  # ms
    n_ap = int(max_shift / dt_ms) + 1
    for i0 in range(nt_tr):
        for j0 in range(1, ns):
            t0 = t[j0]
            for i in range(nt_tr):
                dx = abs(x[i] - x[i0])
                if dx > aperture_m:
                    continue
                tx = np.sqrt(t0 ** 2 + (dx / v_mig * 1000.0) ** 2)
                j = int(round(tx / dt_ms))
                if 0 <= j < ns:
                    out[i0, j0] += data[i, j] / (1 + (dx / aperture_m) ** 2)
    return (
        f"绕射叠加偏移完成: {nt_tr} 道 x {ns} 采样 (dx={dx_m}m, v={v_mig}m/s, "
        f"孔径 {aperture_m:.0f}m≈{n_ap}采样)\n"
        f"输出能量: 输入 rms={np.sqrt((data**2).mean()):.4g} → "
        f"偏移后 rms={np.sqrt((out**2).mean()):.4g}\n"
        f"偏移剖面(前2道各前15采样): {np.round(out[:2, :15], 4).tolist()}"
    )
