# Changelog

## v0.7.1 (2026-09-06)

桌面版界面重设计 + 品牌图标。

- 全新聊天界面（对齐主流 AI 客户端）：左侧边栏（logo/新对话/工作区/保存会话/
  评测入口/会话列表/在线状态）、用户右·助手左消息气泡、底部输入区
- GUI exe 改为目录版（--onedir）：启动从 ~40s 降到 ~5s，修复"双击打不开"的观感
- 艺术大写 R 图标：渐变 R + 地震波纹（assets/ricardo.ico），窗口/任务栏/桌面快捷方式统一
- exe 内嵌图标（--icon），窗口图标经 --add-data 随包分发

## v0.7.0 (2026-09-05)

深度学习 + 处理链 + 体验大版本（56 个工具，22 项测试）。

- PyTorch 工作流：`train_mlp_classifier`（标准化/早停/损失曲线/保存最优权重）、
  `mlp_predict`（加载预测+概率）；表格数据地震属性→岩性分类开箱即用
- 地震处理链：`nmo_velocity_scan`（semblance 速度谱，合成数据反演精度 <1%）、
  `nmo_correct`（双曲线动校正+拉伸切除）、`stack_traces`、`diffraction_stack_migrate`
- 绘图深化：`plot_spectrogram`（时频谱）、`plot_amplitude_section`（彩色剖面）、
  `plot_three_component`（三分量）；所有 plot 工具出图后自动视觉质检
  （需密钥，RICARDO_AUTO_VISION=0 关闭）
- RAG 深化：`lit_ingest_crossref`（Crossref 检索→摘要入库）、
  `lit_compare`（多篇文献联合对比表）
- Agent 自评测基准：`ricardo bench`（7 个标准场景，期望工具调用回归检查）
- Token 用量：`/usage` 命令 + GUI 每轮回复后自动显示
- 双 exe：RicardoAgent.exe（终端）+ RicardoAgentGUI.exe（无控制台黑窗）

## v0.6.1 (2026-09-05)

语义检索升级（45 个工具）。

- 三级检索后端，按可用性自动选择：provider（当前提供商 /embeddings 接口，
  可用 GEOAGENT_EMBED_MODEL 指定模型）→ local（sentence-transformers，
  RICARDO_LOCAL_EMBED_MODEL）→ tfidf（纯本地兜底）
- `lit_reindex`：一键重建语义索引；lit_ingest 增量补向量，后端不可用时自动降级
- lit_search 显式指定语义后端但库为 TF-IDF 时，现场懒升级构建向量
- 同义改述命中：中文查询可直接命中英文段落（向量化后余弦匹配）
- 各提供商默认嵌入模型映射（openai/zhipu/qwen/siliconflow/ollama）
- mock 嵌入端到端测试：中文查询→英文段落 命中率 1.0；16→17 项测试通过

## v0.6.0 (2026-09-05)

文献 RAG（44 个工具）。

- `arxiv_search`：arXiv 检索（支持 cat:physics.geo-ph 等语法）
- `arxiv_download`：下载论文 PDF 并自动入库
- `lit_ingest`：PDF/TXT/MD（含目录递归）解析入库，段落+句子混合分块（约 1200 字符、重叠 200，带页码）
- `lit_search`：本地 TF-IDF 余弦检索（纯 numpy，中文按字 bigram，离线可用）
- `lit_ask`：RAG 问答——检索相关段落 + LLM 综合回答并标注来源编号
- `lit_status`：文献库状态
- 索引按工作区隔离（<工作区>/.ricardo_lit/），每个课题一套文献库
- 新增依赖分组：[rag]（pypdf）

## v0.5.0 (2026-09-05)

体验与韧性大版本。

- 流式输出：CLI/GUI 打字机效果，长回答不再干等（`chat(..., on_event=)`）
- 工具调用可视化：`🔧 工具名(参数)` 与结果摘要在 CLI/GUI 实时显示，对话不再黑盒
- API 韧性：网络/接口错误指数退避自动重试（最多 3 次，未产生输出前才重试）
- `/model` 无参数时自动拉取提供商远端模型列表（失败回退内置预设）；GUI 切换提供商后台刷新
- GUI 离线简单模式：无密钥也能用关键词指令
- 会话快照：`/save` `/load` `/sessions`（存 ~/.geoagent/sessions/，跨天继续课题）
- 独立 exe：`build_exe.cmd` → dist/RicardoAgent.exe（约 142MB，双击即用，含绘图）
- run_python 安全说明明确化（本机权限真实执行，仅超时与工作目录约束）

## v0.4.3 (2026-09-05)

视觉自查（多模态）。

- `see_image`：把工作区图片（PNG/JPG/WebP/GIF）编码后交给多模态模型，
  按科研图件质检维度（标签/单位/图例/配色/数据异常）返回结论与问题清单
- 出图 → 看图 → 修正的自检闭环：system prompt 约定 plot_* 后必须 see_image 自查
- 可用 GEOAGENT_VISION_MODEL 指定专门视觉模型（默认沿用当前会话模型）
- 离线/路径越界/格式/大小守卫完备，13 项测试全部通过

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
