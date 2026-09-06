# 贡献指南（Contributing to Ricardo Agent）

感谢参与！Ricardo Agent 遵循「模型可以换，工具是你的资产」的理念，欢迎贡献领域工具与工程改进。

## 开发环境

```bash
git clone https://github.com/wangRicardo/geoagent.git
cd geoagent
pip install -e ".[all,dev]"
```

## 开发流程

1. 从 `main` 拉功能分支：`git checkout -b feat/my-tool`
2. 开发 + 补测试（`tests/`）
3. 本地质量门（与 CI 完全一致）：
   ```bash
   ruff check geoagent tests cli.py     # lint，必须零违规
   ruff format geoagent tests cli.py    # 格式
   python -m pytest tests/ -q           # 测试，必须全过
   ```
4. 提交并推送，开 Pull Request。CI 会跑 lint + 三平台三版本测试矩阵 + wheel 构建。

## 添加领域工具的规范

- 函数放 `geoagent/tools/` 对应模块，必须带类型注解和 docstring（首段会成为 LLM 的工具描述）
- 重依赖一律懒导入 + 缺失时返回 `ERROR: ... 安装指引` 文本，不允许让 agent 崩溃
- 所有文件读写限制在工作目录内（参考 `fs_tools._safe_path`）
- 新工具要有测试；涉及网络的工具测试需能在离线环境跳过或走 mock
- 有明确类别：geophysics / processing / ml / plot / data / rag / latex / files / memory / vision / citation

## 场景基准

改动 `core.py` 或 system prompt 后，请运行 `ricardo bench`（需在线），
或在 `benchmarks/scenarios.json` 补充你的场景——工具调用行为是回归保护的重点。

## 发布

维护者流程：更新 `geoagent/__init__.py` 的 `__version__` 与 CHANGELOG → 打 tag `vX.Y.Z` 推送 →
GitHub Actions 自动构建 wheel/双 exe 并发布 Release（PyPI 发布需在仓库 Variables 配置 `PUBLISH_PYPI=true` 及 Trusted Publisher）。

## 行为准则

保持友善、专业；讨论对事不对人。
