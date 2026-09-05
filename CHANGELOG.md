# Changelog

## v0.2.0 (2026-09-05)

首个公开发布版本。

- Agent 核心：模型无关的 function-calling 循环（任何 OpenAI 兼容接口）+ 离线模式
- 地球物理工具：SEG-Y 读取、LAS 测井、FFT 频谱、Butterworth 滤波、TWT→深度
- 地震正演：Ricker 子波、1D 褶积合成记录、调谐厚度/分辨率估算
- 机器学习工具：数据画像、随机森林基线、k 折交叉验证、PCA、特征相关性、混淆矩阵报告
- Agent 记忆：研究笔记的持久化保存/检索（~/.geoagent/notes.json）
- CLI：`geoagent tools|run|chat`
- 可选依赖分组安装（`pip install geoagent-ml[all]`）
