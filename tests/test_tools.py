"""工具层冒烟测试（不依赖重型库即可跑核心路径）。"""

import math

import numpy as np

from geoagent import registry


def test_tools_registered():
    names = {t.name for t in registry.list()}
    assert {
        "signal_spectrum",
        "signal_filter",
        "dataset_profile",
        "segy_info",
        "las_curves",
        "ricker_wavelet",
        "synthetic_trace",
        "save_note",
        "feature_correlation",
    } <= names


def test_ricker_and_synthetic():
    out = registry.execute("ricker_wavelet", {"freq_hz": 30.0})
    assert "零相位" in out
    out = registry.execute(
        "synthetic_trace", {"impedance": [2000, 2000, 4000, 2000, 2000], "wavelet_freq_hz": 30.0}
    )
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
    out = registry.execute("velocity_to_depth", {"vp_ms": 3000.0, "two_way_time_ms": [1000.0, 2000.0]})
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
    registry.execute(
        "latex_write", {"filename": "bad.tex", "content": "\\documentclass{article}\begin{document}\broken"}
    )
    assert "ERROR" in registry.execute(
        "latex_compile", {"filename": "bad.tex", "engine": "pdflatex", "passes": 1}
    )


def test_plot_tools(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(1)
    t = np.arange(200) / 50.0
    seis = [(np.sin(2 * np.pi * 8 * t) + rng.normal(0, 0.05, 200)).tolist() for _ in range(10)]
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
    import segyio  # noqa: F401

    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(3)
    t = np.arange(200) * 0.002
    traces = [
        (np.sin(2 * np.pi * 20 * t) * np.exp(-3 * t) + rng.normal(0, 0.02, 200)).tolist() for _ in range(12)
    ]
    out = registry.execute("segy_write", {"traces": traces, "dt_ms": 2.0, "filename": "syn.sgy"})
    assert "已写入" in out
    info = registry.execute("segy_info", {"path": "syn.sgy"})
    assert "12" in info and "采样" in info
    amp = registry.execute("segy_trace_amplitude", {"path": "syn.sgy", "trace_index": 0})
    assert "rms=" in amp
    win = registry.execute(
        "segy_extract_window", {"path": "syn.sgy", "trace_index": 0, "t_start_ms": 100, "t_end_ms": 200}
    )
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


def test_see_image_guards(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # 无密钥时的明确提示
    out = registry.execute("see_image", {"path": "a.png"})
    assert "ERROR" in out and ("密钥" in out or "offline" in out)
    # 路径越界与格式守卫（有密钥环境也不该崩溃）
    monkeypatch.setenv("RICARDO_API_KEY", "sk-test")
    out = registry.execute("see_image", {"path": "../x.png"})
    assert "越出工作目录" in out
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    out = registry.execute("see_image", {"path": "a.txt"})
    assert "不支持的图片格式" in out
    out = registry.execute("see_image", {"path": "missing.png"})
    assert "不存在" in out


def test_session_save_load(tmp_path, monkeypatch):
    from geoagent import GeoAgent
    from geoagent import core as core_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(core_mod, "SESSIONS_DIR", tmp_path / "sessions")
    agent = GeoAgent()
    agent.history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    assert "已保存" in agent.save_session("projA")
    agent.history = []
    out = agent.load_session("projA")
    assert "已恢复" in out and agent.history[-1]["content"] == "a"
    assert "projA" in agent.list_sessions()
    assert "不存在" in agent.load_session("nope")


def test_offline_chat_events(tmp_path, monkeypatch):
    from geoagent import GeoAgent

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RICARDO_API_KEY", raising=False)
    monkeypatch.delenv("GEOAGENT_API_KEY", raising=False)
    agent = GeoAgent()
    events = []
    out = agent.chat("hi", on_event=events.append)
    assert "offline" in out and events == []  # 离线路径不产生流事件


def test_lit_rag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paper = "\n\n".join(
        f"Section {i}: Ambient noise tomography uses surface wave dispersion "
        f"curves from seismic interferometry. Phase velocity maps at {i}s period "
        f"reveal crustal structure beneath the array. " * 3
        for i in range(1, 15)
    )
    (tmp_path / "paper.txt").write_text(paper, encoding="utf-8")
    out = registry.execute("lit_ingest", {"paths": ["."]})
    assert "入库完成" in out
    assert "paper.txt" in registry.execute("lit_status", {})
    out = registry.execute("lit_search", {"query": "ambient noise surface wave dispersion", "k": 3})
    assert "[0." in out
    # 离线 ask：给出检索结果并提示需要密钥
    out = registry.execute("lit_ask", {"question": "什么是 ambient noise tomography?"})
    assert "ERROR" in out and "密钥" in out
    out = registry.execute("arxiv_download", {"arxiv_id": "bad-id"})
    assert "ERROR" in out


def test_lit_semantic_backend(tmp_path, monkeypatch):
    import geoagent.tools.lit_tools as lit

    monkeypatch.chdir(tmp_path)
    (tmp_path / "paper.txt").write_text(
        "\n\n".join(
            f"Deep learning phase picking with neural networks on seismic "
            f"waveforms, experiment {i}: transformer encoder attends to "
            f"multi-station features for earthquake detection. " * 4
            for i in range(1, 8)
        ),
        encoding="utf-8",
    )
    assert "入库完成" in registry.execute("lit_ingest", {"paths": ["paper.txt"]})
    assert "tfidf" in registry.execute("lit_status", {})

    # mock 一个本地嵌入后端：固定 4 维语义向量（同义改述向量相近）
    calls = []

    def fake_local(texts, model=None):
        calls.append(len(texts))
        vec_map = {
            "机器学习": [1, 1, 0, 0],
            "deep learning": [1, 1, 0, 0],
            "拾震相": [0, 0, 1, 1],
            "phase picking": [0, 0, 1, 1],
        }
        out = []
        for t in texts:
            tl = t.lower()
            v = [0.0, 0.0, 0.0, 0.0]
            for k, vec in vec_map.items():
                if k in tl:
                    v = [a + b for a, b in zip(v, vec, strict=False)]
            out.append(v if any(v) else [0.1, 0.1, 0.1, 0.1])
        arr = np.asarray(out, dtype=np.float32)
        return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)

    monkeypatch.setattr(lit, "_embed_local", fake_local)

    # 中文查询命中英文段落（语义检索的核心价值）
    out = registry.execute("lit_search", {"query": "机器学习 拾震相", "k": 2, "backend": "local"})
    assert "后端 local" in out, out
    assert "[1." in out
    assert calls  # 确实走了嵌入
    # lit_reindex 本地后端重建
    out = registry.execute("lit_reindex", {"backend": "local"})
    assert "语义索引重建完成" in out, out
    # provider 后端不可用时优雅回退
    monkeypatch.delenv("RICARDO_API_KEY", raising=False)
    monkeypatch.delenv("GEOAGENT_API_KEY", raising=False)
    out = registry.execute("lit_reindex", {"backend": "provider"})
    assert "ERROR" in out
    # tfidf 兜底依然可用
    out = registry.execute("lit_search", {"query": "transformer encoder", "k": 1})
    assert "tfidf" in out or "后端" in out


def test_processing_chain(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    dt, n = 2.0, 800
    offsets = np.linspace(100, 2400, 24).tolist()
    data = []
    for x in offsets:
        tr = rng.normal(0, 0.05, n)
        for t0, v in [(500, 2000), (1000, 3500)]:
            j = int(np.sqrt(t0**2 + (x / v * 1000) ** 2) / dt)
            if j < n:
                tr[j] += 1.0
        data.append(tr.tolist())
    picks = registry.execute(
        "nmo_velocity_scan", {"traces": data, "dt_ms": dt, "offsets_m": offsets, "n_picks": 2}
    )
    assert "速度谱拾取" in picks
    assert "(494.0, 2008" in picks and "(990.0, 3534" in picks  # 反演精度 <1%
    assert "NMO 校正完成" in registry.execute(
        "nmo_correct", {"traces": data, "dt_ms": dt, "offsets_m": offsets, "velocity": 2000.0}
    )
    assert "叠加完成" in registry.execute("stack_traces", {"traces": data})
    assert "绕射叠加偏移完成" in registry.execute(
        "diffraction_stack_migrate", {"traces": data[:8], "dx_m": 100.0}
    )


def test_mlp_train_predict(tmp_path, monkeypatch):
    pytest_import_or_skip()
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (120, 3)).tolist()
    y = [1 if (r[0] + r[1]) > 0.5 else 0 for r in X]
    out = registry.execute(
        "train_mlp_classifier", {"features": X, "labels": y, "epochs": 8, "model_name": "t1"}
    )
    assert "训练完成" in out and "验证集准确率" in out
    assert (tmp_path / "ricardo_models" / "t1.json").exists()
    out = registry.execute("mlp_predict", {"model_name": "t1", "features": X[:2]})
    assert "→" in out


def pytest_import_or_skip():
    try:
        import torch  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("未安装 torch")
    return torch


def test_new_plots(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    t = np.arange(500) * 0.002
    sig = (np.sin(2 * np.pi * 20 * t)).tolist()
    for name, kwargs in [
        ("plot_spectrogram", {"data": sig, "dt_ms": 2.0}),
        ("plot_amplitude_section", {"traces": [sig, list(reversed(sig))], "dt_ms": 2.0}),
        ("plot_three_component", {"data_e": sig, "data_n": sig, "data_z": sig}),
    ]:
        out = registry.execute(name, kwargs)
        assert "已保存" in out, out


def test_lit_compare_needs_multiple(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("ambient noise tomography study one", encoding="utf-8")
    registry.execute("lit_ingest", {"paths": ["a.txt"]})
    out = registry.execute("lit_compare", {"question": "结论是什么?"})
    assert "ERROR" in out or "至少需要 2 篇" in out


def test_benchmark_harness_offline():
    from geoagent.bench import load_scenarios, run_benchmark

    class FakeAgent:
        online = True

        def chat(self, user, on_event=None):
            on_event({"type": "tool_start", "name": "ricker_wavelet"})
            return "ok"

    sc = [s for s in load_scenarios() if s["name"] == "ricker_forward"]
    assert "✅" in run_benchmark(FakeAgent(), sc)

    class OfflineAgent:
        online = False

    assert "ERROR" in run_benchmark(OfflineAgent(), sc)
