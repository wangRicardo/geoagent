# 安全策略（Security Policy）

## 支持版本

| 版本 | 支持 |
|---|---|
| latest release | ✅ |
| 更早版本 | ❌（请升级） |

## 报告漏洞

请勿公开提交安全漏洞。通过 GitHub Security Advisories（仓库 Security 标签 → Report a vulnerability）
私下报告，我们会在 72 小时内确认、7 天内给出修复或缓解方案。

## 已知设计边界（使用前必读）

1. **`run_python` 无沙箱**：该工具在工作目录内以当前用户权限真实执行任意 Python 代码。
   仅有的约束是超时与进程隔离。请只让 agent 运行你审阅过的代码；
   不要把含敏感信息的环境变量暴露给不可信来源生成的内容。
2. **文件工具目录隔离**：`read_file` / `write_file` / `list_files` / `latex_*` / `lit_*`
   均限制在当前工作目录内，路径越界（`..`、绝对路径逃逸）会被拒绝。
3. **API 密钥**：仅从环境变量读取（按提供商命名或通用 `RICARDO_API_KEY`），
   不落盘、不进配置文件、不写入对话历史。
4. **网络工具**（USGS / IRIS / arXiv / Crossref / 嵌入接口）为只读查询；
   下载的文件先落在工作区，PDF 解析用纯 Python（pypdf），不执行文档内容。

## 数据本地性

研究笔记（`~/.geoagent/notes.json`）、会话快照（`~/.geoagent/sessions/`）、
日志（`~/.geoagent/logs/`）与文献索引（工作区 `.ricardo_lit/`）全部保存在本地，
除你主动调用的 LLM/检索 API 外没有遥测与上传。
