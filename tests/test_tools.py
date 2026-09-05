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


def test_latex_tools(tmp_path, monkeypatch):
    import shutil
    if not shutil.which("xelatex"):
        import pytest
        pytest.skip("本机无 LaTeX 引擎")
    monkeypatch.chdir(tmp_path)
    assert "xelatex" in registry.execute("latex_check", {})
    assert "已写入" in registry.execute("latex_write", {"filename": "t.tex", "title": "测试"})
    out = registry.execute("latex_compile", {"filename": "t.tex"})
    assert "编译成功" in out and (tmp_path / "t.pdf").exists()
    registry.execute("latex_write", {"filename": "bad.tex",
                                     "content": "\documentclass{article}\begin{document}\broken"})
    assert "ERROR" in registry.execute("latex_compile", {"filename": "bad.tex", "engine": "pdflatex", "passes": 1})


def test_plot_tools(tmp_path, monkeypatch):
    matplotlib = __import__("matplotlib")
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(1)
    t = np.arange(200) / 50.0
    seis = [(np.sin(2 * np.pi * 8 * t) + rng.normal(0, .05, 200)).tolist() for _ in range(10)]
    for name, kwargs in [
        ("plot_seismic_section", {"traces": seis, "dt_ms": 20}),
        ("plot_crossplot", {"x": rng.normal(0, 1, 50).tolist(), "y": rng.normal(0, 1, 50).tolist()}),
        ("plot_well_logs", {"curves": [rng.normal(80, 10, 50).tolist()], "names": ["GR"]}),
        ("plot_time_series", {"series": [np.sin(2 * np.pi * 5 * t).tolist()], "labels": ["s"]}),
    ]:
        out = registry.execute(name, kwargs)
        assert "已保存" in out, out
    assert (tmp_path / "seismic_section.png").exists()
    assert (tmp_path / "crossplot.png").exists()


def test_citation_lookup_offline_error():
    # 网络不可用时必须返回 ERROR 文本而不是抛异常
    out = registry.execute("citation_lookup", {"query": "nonexistent-query-xyz", "rows": 1})
    assert out.startswith("ERROR") or "没有找到" in out or "DOI" in out


def test_segy_write_read_roundtrip(tmp_path, monkeypatch):
    segyio = __import__("segyio")
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(3)
    t = np.arange(200) * 0.002
    traces = [(np.sin(2 * np.pi * 20 * t) * np.exp(-3 * t) + rng.normal(0, .02, 200)).tolist()
              for _ in range(12)]
    out = registry.execute("segy_write", {"traces": traces, "dt_ms": 2.0, "filename": "syn.sgy"})
    assert "已写入" in out
    info = registry.execute("segy_info", {"path": "syn.sgy"})
    assert "12" in info and "采样" in info
    amp = registry.execute("segy_trace_amplitude", {"path": "syn.sgy", "trace_index": 0})
    assert "rms=" in amp
    win = registry.execute("segy_extract_window",
                           {"path": "syn.sgy", "trace_index": 0, "t_start_ms": 100, "t_end_ms": 200})
    assert "个采样" in win


def test_history_trim_and_export(tmp_path, monkeypatch):
    from geoagent import GeoAgent
    monkeypatch.chdir(tmp_path)
    agent = GeoAgent()
    agent.max_history = 4
    for i in range(6):
        agent.history.append({"role": "user", "content": f"q{i}"})
        agent.history.append({"role": "assistant", "content": f"a{i}"})
    agent._trim_history()
    assert len(agent.history) == 4
    assert agent.history[0]["content"] == "q4"  # 保留最近 4 条，从 user 轮开始
    out = agent.export_transcript("log.md")
    assert "已导出" in out
    text = (tmp_path / "log.md").read_text(encoding="utf-8")
    assert "a5" in text and "Ricardo" in text
