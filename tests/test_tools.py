"""工具层冒烟测试（不依赖重型库即可跑核心路径）。"""

import math

import numpy as np

from geoagent import registry


def test_tools_registered():
    names = {t.name for t in registry.list()}
    assert {"signal_spectrum", "signal_filter", "dataset_profile",
            "segy_info", "las_curves", "ricker_wavelet", "synthetic_trace",
            "save_note", "feature_correlation"} <= names


def test_ricker_and_synthetic():
    out = registry.execute("ricker_wavelet", {"freq_hz": 30.0})
    assert "零相位" in out
    out = registry.execute("synthetic_trace",
                           {"impedance": [2000, 2000, 4000, 2000, 2000],
                            "wavelet_freq_hz": 30.0})
    assert "合成地震道" in out and "反射系数" in out


def test_notes_roundtrip(tmp_path, monkeypatch):
    from geoagent.tools import notes as notes_mod
    monkeypatch.setattr(notes_mod, "NOTES_PATH", tmp_path / "notes.json")
    registry.execute("save_note", {"title": "测试", "content": "调谐厚度结论", "tags": "test"})
    assert "测试" in registry.execute("list_notes", {})
    assert "调谐厚度" in registry.execute("search_notes", {"keyword": "调谐"})
    out = registry.execute("read_note", {"note_id": 1})
    assert "测试" in out


def test_signal_spectrum():
    dt_ms = 1.0
    t = np.arange(0, 1000) * dt_ms / 1000.0
    x = (2 * np.sin(2 * math.pi * 10 * t)).tolist()
    out = registry.execute("signal_spectrum", {"data": x, "dt_ms": dt_ms})
    assert "10.0" in out  # 主频应为 10 Hz


def test_dataset_profile_and_quick_train():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3)).tolist()
    y = [1 if (r[0] + r[1]) > 0 else 0 for r in X]
    out = registry.execute("dataset_profile", {"features": X, "target": y})
    assert "100 样本" in out and "y 取值分布" in out
    out = registry.execute("quick_train", {"features": X, "target": y})
    assert "classification" in out or "scikit-learn" in out


def test_unknown_tool_returns_error_text():
    out = registry.execute("no_such_tool", {})
    assert out.startswith("ERROR")


def test_velocity_to_depth():
    out = registry.execute("velocity_to_depth",
                           {"vp_ms": 3000.0, "two_way_time_ms": [1000.0, 2000.0]})
    assert "1500.0" in out and "3000.0" in out
