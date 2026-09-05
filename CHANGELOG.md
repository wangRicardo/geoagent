# Changelog

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
