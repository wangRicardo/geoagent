# Changelog

## v0.4.2 (2026-09-05)

功能审核与补全（37 个工具）。

- `segy_write`：多道地震数据写入 SEG-Y（IEEE float32，segyio 兼容，时间轴已验证）
- `segy_extract_window`：指定道时窗振幅提取
- 对话上下文自动裁剪（默认保留最近 40 条，成对丢弃防撑爆上下文）
- `/export` 命令（CLI + GUI）：对话导出为 Markdown 研究日志
- GUI 支持全部斜杠命令（/cd /model /provider /thinking /export 等）
- 新增 SEGY 读写回环与历史裁剪测试（12 项全部通过）

## v0.4.1 (2026-09-05)

桌面版本发布（`ricardo gui` / `ricardo-gui`）。

- Tkinter 实现，零额外依赖；深色主题，与官网同风格
- 顶栏：工作区文件夹选择（运行中随时切换）、提供商/模型/思考强度下拉热切换（持久化到 config.json）
- 聊天区分色显示系统消息/用户/助手/错误；LLM 调用在后台线程，界面不卡
- 未配置密钥时进入 offline 模式并给出配置指引

## v0.4.0 (2026-09-05)

科研全流程补全（35 个工具）。

- 数据获取：`fetch_usgs_earthquakes`（USGS 地震目录）、`fetch_iris_events`（全球事件，USGS FDSN）、`fetch_iris_waveform`（IRIS 波形下载为 SAC）
- 图片生成：`plot_seismic_section`（变面积剖面）、`plot_crossplot`（交会图+相关系数）、`plot_well_logs`（多曲线并排）、`plot_time_series`（时序叠加）
- 文献：`citation_lookup`（Crossref 检索/DOI 精确查询 → BibTeX，可追加保存 .bib）
- 新增依赖分组：`pip install "geoagent-ml[data]"`（obspy, requests）、`[plot]`（matplotlib）

## v0.3.1 (2026-09-05)

- LaTeX 工具链：latex_check / latex_write（中文论文模板）/ latex_compile（xelatex 两遍，自动识别 ctex，错误日志提取）

## v0.3.0 (2026-09-05)

更名为 **Ricardo Agent**。

- 任意文件夹工作：`ricardo chat <文件夹>` 或对话内 `/cd 路径`，类似 codex / claude code 的工作区
- 文件系统工具：`list_files` / `read_file` / `write_file` / `run_python`（限制在工作目录内）
- 多提供商：内置 openai / deepseek / zhipu / moonshot / qwen / ollama 预设，持久化到 `~/.geoagent/config.json`
- 对话内热切换：`/provider` `/model` `/thinking`（off/low/medium/high → reasoning_effort）/`/clear`
- CLI 启动横幅（ASCII LOGO）与状态栏（提供商/模型/思考强度/在线状态/工作目录）

## v0.2.0 (2026-09-05)

首个公开发布版本。

- Agent 核心：模型无关的 function-calling 循环（任何 OpenAI 兼容接口）+ 离线模式
- 地球物理工具：SEG-Y 读取、LAS 测井、FFT 频谱、Butterworth 滤波、TWT→深度
- 地震正演：Ricker 子波、1D 褶积合成记录、调谐厚度/分辨率估算
- 机器学习工具：数据画像、随机森林基线、k 折交叉验证、PCA、特征相关性、混淆矩阵报告
- Agent 记忆：研究笔记的持久化保存/检索（~/.geoagent/notes.json）
- CLI：`geoagent tools|run|chat`
- 可选依赖分组安装（`pip install geoagent-ml[all]`）
