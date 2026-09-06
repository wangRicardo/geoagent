# Ricardo Agent —— 你的地球物理 × 机器学习个人 Agent

[![CI](https://github.com/wangRicardo/geoagent/actions/workflows/ci.yml/badge.svg)](https://github.com/wangRicardo/geoagent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-22d3ee.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/geoagent-ml?color=818cf8)](https://pypi.org/project/geoagent-ml/)
[![Python](https://img.shields.io/badge/Python-3.10+-34d399.svg)](pyproject.toml)

一个从零开始、可自己持续扩展的研究 agent 框架（当前版本 **v0.3.0**，内置 58 个工具，可在任意文件夹工作）。核心理念：**模型可以换，工具是你的资产**。
agent 的智力来自 LLM，价值来自你亲手写的领域工具（读 SEG-Y、看测井曲线、快速建模……）。

官网：<https://wangricardo.github.io/geoagent/>

安全须知见 [SECURITY.md](SECURITY.md)；参与贡献见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 安装

```bash
pip install geoagent-ml          # 核心（numpy）
pip install "geoagent-ml[all]"   # 含 scikit-learn / scipy / segyio / lasio / openai
```

## 快速开始

```bash
# 离线使用（无需 API key）
python cli.py tools                 # 列出所有工具
python cli.py run signal_spectrum "data=[1,2,3,2,1]; dt_ms=1.0"
python examples/demo.py             # 迷你研究流程演示

# 在线对话（任意 OpenAI 兼容接口）
pip install openai
set GEOAGENT_API_KEY=sk-...          # Windows: set / $env: in PowerShell
set GEOAGENT_BASE_URL=https://api.openai.com/v1   # 可换成任何兼容端点
set GEOAGENT_MODEL=gpt-4o-mini
python cli.py chat
```

## 项目结构

```
geoagent/
  core.py            # agent 循环：system prompt + function calling 工具循环
  tools/
    base.py          # 工具注册表：函数 + docstring 自动变成 function-calling schema
    geophysics.py    # 地球物理工具
    ml_tools.py      # 机器学习工具
tests/test_tools.py  # 工具层测试（python -m pytest tests/）
examples/demo.py     # 合成数据端到端演示
cli.py               # 命令行入口
```

## 内置工具

| 类别 | 工具 | 说明 |
|---|---|---|
| 地球物理 | `segy_info` / `segy_trace_amplitude` | SEG-Y 文件信息与道数据（需 `segyio`） |
| 地球物理 | `las_curves` / `las_read_curve` | LAS 测井曲线浏览（需 `lasio`） |
| 地球物理 | `signal_spectrum` / `signal_filter` | FFT 频谱、Butterworth 滤波（numpy/scipy） |
| 地球物理 | `velocity_to_depth` | TWT→深度转换 |
| 机器学习 | `dataset_profile` | 数据画像：形状/统计/缺失/类别分布 |
| 机器学习 | `quick_train` | 自动分类/回归基线（随机森林）+ 指标 + 特征重要性 |
| 机器学习 | `cross_validate` | k 折交叉验证 |
| 机器学习 | `pca_reduce` | PCA 降维 |

重型依赖（segyio/lasio/scikit-learn）都是可选的：没装时工具返回安装提示，agent 本身不崩。

## 如何加自己的工具（这是本项目的主要玩法）

在 `geoagent/tools/` 任意文件里写一个带类型注解和 docstring 的函数，然后注册：

```python
from .base import registry

@registry.register(category="geophysics")
def wavelet_ricker(freq_hz: float, dt_ms: float = 1.0, length_ms: float = 100.0) -> str:
    """生成 Ricker 子波并返回采样值，可叠加到合成地震记录实验中。"""
    ...
```

函数名、docstring 第一段、类型注解会自动进入 function-calling schema，
在线模式下 LLM 立刻就能调用它 —— 不需要写任何额外胶水代码。

## 建议的扩展路线图

1. **数据接入**：`pick` 工具（读取拾取文件）、HDF5/NetCDF 地震体切片、VSP/FKO 数据。
2. **正演与反演**：Ricker 子波 + 一维褶积模型制作合成记录；速度分析工具。
3. **ML 深化**：XGBoost/LightGBM、PyTorch 训练封装（断点、早停）、地震相分类流水线。
4. **可视化**：matplotlib 保存工区剖面/交会图 PNG，让 agent 能"看图说话"（配合多模态模型）。
5. **Agent 能力**：多步规划（把"解释这口井"拆成工具序列）、检索文献（RAG）、长期记忆（研究笔记 JSON）。
6. **评测**：为 agent 建 benchmark —— 一组标准研究问题 + 期望工具调用序列，防止改坏。

## 设计说明

- 工具结果一律转成文本回传 LLM（`ToolRegistry.execute`），方便任何模型接入与调试。
- 工具内部异常被捕获为 `ERROR: ...` 文本返回，让模型自行纠正参数而不是崩溃。
- `max_tool_rounds` 限制单轮对话的工具调用次数，防止失控循环。

## 桌面版（v0.4.1）

```bash
ricardo gui              # 图形界面：文件夹工作区 + 提供商/模型/思考强度下拉热切换
```

## 厂商管理与权限模式（v0.9.0）

- **软件内设置**：⚙ 设置窗口添加多个自定义厂商（地址+密钥+默认模型）、保存/切换/测试连接，多厂商并存随时换模型
- **权限模式**：只读 / 标准 / 谨慎（逐次确认）/ 自主 四档，按工具风险分级（联网/执行/写入/读取）拦截，CLI `/mode` 或界面下拉切换

## 深度学习与处理链（v0.7.0）

- PyTorch MLP 训练（早停+损失曲线）与预测：`train_mlp_classifier` / `mlp_predict`（`pip install torch --index-url https://download.pytorch.org/whl/cpu`）
- 地震处理链：semblance 速度扫描 → NMO → 叠加 → 绕射偏移
- 新绘图：时频谱图、彩色振幅剖面、三分量地震图；出图自动视觉质检
- `ricardo bench`：agent 行为回归评测（7 场景）
- `/usage` token 统计；双 exe 构建（终端 + 无黑窗 GUI）

## 文献 RAG（v0.6.0）

文献研究闭环：`arxiv_search` 检索 → `arxiv_download` 下载 → `lit_ingest` 入库 → `lit_search` 定位原文 → `lit_ask` 带引用回答。
检索三级后端自动选择：**provider**（提供商 /embeddings，`GEOAGENT_EMBED_MODEL` 可指定模型）→ **local**（sentence-transformers）→ **tfidf**（离线兜底）。`lit_reindex` 一键重建；中文查询可直接命中英文段落。索引随工作区隔离（`.ricardo_lit/`）。

## 体验特性（v0.5.0）

- 流式输出 + 工具调用实时可视化（CLI/GUI）
- API 错误自动重试（指数退避）
- `/save` `/load` 会话快照；`/model` 自动拉取提供商真实模型列表
- 独立 exe：运行 `build_exe.cmd` 生成 `dist/RicardoAgent.exe`，无需 Python 环境
- ⚠️ 安全说明：`run_python` 在本机以当前用户权限真实执行，只运行你审阅过的代码

## 视觉自查（v0.4.3）

- `see_image`：agent 生成图件后自动"看图"质检——坐标轴、图例、配色、数据异常，
  发现问题自动修正重绘（需多模态模型，如 gpt-4o；可用 `GEOAGENT_VISION_MODEL` 指定）

## SEG-Y 写入与对话导出（v0.4.2）

- `segy_write`：把合成记录/处理后的道集写成标准 SEG-Y（Petrel 等软件可读）
- `segy_extract_window`：指定道时窗振幅提取
- 对话历史自动裁剪 + `/export` 导出 Markdown 研究日志（CLI 与 GUI 均支持）

## 数据与图件（v0.4.0）

- 数据下载：USGS 地震目录、IRIS 波形（SAC）、全球事件查询（需 `pip install obspy`）
- 图片生成：地震剖面变面积图、交会图（自带相关系数）、测井曲线并排图、时序对比图（自动保存 PNG 到工作区）
- 文献：Crossref 检索/DOI 查询，直接生成 BibTeX 可存为 .bib 文件

## LaTeX 支持（v0.3.1）

- `latex_check`：检测本机 TeX 发行版（MiKTeX / TeX Live / tectonic）与各引擎版本
- `latex_write`：写入 .tex（内置支持中文的 ctexart 论文模板，可自定义内容）
- `latex_compile`：编译为 PDF，默认 xelatex 两遍（自动识别 ctex 文档类），失败时返回日志错误行

## v0.3.0 新特性

- **任意文件夹工作**：`ricardo chat D:\课题\工区A` 或对话内 `/cd 路径`，agent 的工作区随你切换（类似 codex / claude code）
- **文件系统工具**：list_files / read_file / write_file / run_python，全部限制在工作目录内
- **多提供商**：内置 openai / deepseek / zhipu / moonshot / qwen / ollama 预设
- **对话内热切换**：`/provider deepseek`、`/model deepseek-reasoner`、`/thinking high`（off/low/medium/high），配置持久化到 `~/.geoagent/config.json`
- API 密钥按提供商环境变量自动识别（DEEPSEEK_API_KEY 等），或统一用 `RICARDO_API_KEY`
