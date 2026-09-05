"""数据获取工具：地震波形下载（IRIS）与地震目录查询（USGS）。

- fetch_iris_waveform 依赖 obspy（FDSN web service），按台网.台站.台道下载
  指定时间窗的波形并保存为 SAC 文件到工作目录。
- fetch_usgs_earthquakes 只用 requests，查询 USGS 地震目录 API，返回事件列表。
两个工具都会把原始数值返回给 LLM（截断），方便后续直接接 ML/绘图工具。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

import numpy as np

from .base import registry

UA = {"User-Agent": "RicardoAgent/0.4 (research agent; geophysics)"}


@registry.register(category="data")
def fetch_usgs_earthquakes(
    min_magnitude: float = 5.0,
    start_date: str = "2024-01-01",
    end_date: str = "2024-02-01",
    min_latitude: float = -90.0,
    max_latitude: float = 90.0,
    min_longitude: float = -180.0,
    max_longitude: float = -180.0 + 360.0,
    limit: int = 20,
) -> str:
    """查询 USGS 地震目录：返回指定时间窗与矩形范围内的地震事件（时间/震级/位置）。

    日期格式 YYYY-MM-DD。适合做震例筛选、给波形下载选目标事件。
    """
    import requests

    params = {
        "format": "csv",
        "starttime": start_date,
        "endtime": end_date,
        "minmagnitude": min_magnitude,
        "minlatitude": min_latitude,
        "maxlatitude": max_latitude,
        "minlongitude": min_longitude,
        "maxlongitude": max_longitude,
        "limit": min(limit, 200),
        "orderby": "magnitude",
    }
    try:
        r = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query",
                         params=params, headers=UA, timeout=60)
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: USGS 查询失败: {exc}"
    lines = r.text.strip().splitlines()
    if len(lines) <= 1:
        return "查询成功，但没有符合条件的事件。"
    rows = lines[1:][: min(limit, 20)]
    out = [f"共 {len(lines) - 1} 个事件（显示前 {len(rows)} 个，按震级排序）:"]
    for row in rows:
        cols = row.split(",")
        # CSV 列: time,latitude,longitude,depth,mag,magType,nst,gap,dmin,rms,...,place
        out.append(f"  {cols[0][:19]} M{cols[4]} @ ({cols[1]}, {cols[2]}) depth={cols[3]}km | {cols[-1][:50]}")
    return "\n".join(out)


@registry.register(category="data")
def fetch_iris_waveform(
    network: str,
    station: str,
    channel: str = "BHZ",
    location: str = "00",
    start_time: str = "2024-01-01T00:00:00",
    duration_sec: int = 3600,
    filename: Optional[str] = None,
) -> str:
    """从 IRIS FDSN 服务下载指定台站的地震波形，保存为 SAC 文件并返回统计量。

    network/station 为 FDSN 台网与台站代码（如 IU / ANMO）；
    start_time 为 ISO 格式 UTC 时间。依赖 obspy（pip install obspy）。
    """
    try:
        from obspy import UTCDateTime
        from obspy.clients.fdsn import Client
    except ImportError:
        return "ERROR: 需要 obspy。请先安装: pip install obspy"
    try:
        t0 = UTCDateTime(start_time)
    except Exception:
        return "ERROR: start_time 需为 ISO 格式，如 2024-01-01T00:00:00"
    if not filename:
        filename = f"{network}.{station}.{channel}.{t0.strftime('%Y%m%dT%H%M')}.sac"
    try:
        client = Client("IRIS", user_agent=UA["User-Agent"])
        st = client.get_waveforms(network, station, location, channel, t0, t0 + duration_sec)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: IRIS 下载失败: {exc}"
    tr = st.merge()[0]
    tr.data = tr.data.astype(np.float64)
    root = os.getcwd()
    full = os.path.abspath(os.path.join(root, filename))
    if os.path.commonpath([root, full]) != root:
        return f"ERROR: 路径越出工作目录: {filename}"
    st.write(full, format="SAC")
    data = tr.data
    peaks = np.abs(data)
    return (
        f"下载成功: {filename}\n"
        f"  台站: {network}.{station}.{location}.{channel}  "
        f"{tr.stats.starttime} ~ {tr.stats.endtime} (UTC)\n"
        f"  采样率: {tr.stats.sampling_rate} Hz, 样点数: {data.size}\n"
        f"  振幅: min={data.min():.4g}, max={data.max():.4g}, "
        f"rms={np.sqrt(np.mean(data**2)):.4g}, 最大绝对值位置={int(peaks.argmax())}\n"
        f"  前20采样: {np.round(data[:20], 4).tolist()}"
    )


@registry.register(category="data")
def fetch_iris_events(
    min_magnitude: float = 6.0,
    start_date: str = "2024-01-01",
    end_date: str = "2024-02-01",
    limit: int = 10,
) -> str:
    """查询全球地震事件目录（USGS FDSN 端点），作为波形下载的目标清单。"""
    try:
        from obspy import UTCDateTime
        from obspy.clients.fdsn import Client
    except ImportError:
        return "ERROR: 需要 obspy。请先安装: pip install obspy"
    try:
        client = Client("USGS", user_agent=UA["User-Agent"])
        cat = client.get_events(
            starttime=UTCDateTime(start_date), endtime=UTCDateTime(end_date),
            minmagnitude=min_magnitude, orderby="magnitude", limit=min(limit, 50),
        )
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: IRIS 事件查询失败: {exc}"
    out = [f"共 {len(cat)} 个事件:"]
    for ev in cat:
        o = ev.preferred_origin() or ev.origins[0]
        m = ev.preferred_magnitude() or ev.magnitudes[0]
        out.append(
            f"  {o.time.strftime('%Y-%m-%d %H:%M')} M{m.mag:.1f} "
            f"@ ({o.latitude:.2f}, {o.longitude:.2f}) depth={o.depth / 1000:.0f}km"
        )
    return "\n".join(out)
