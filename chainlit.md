## Chainlit + LangChain 工具调用 Demo

欢迎体验本项目的最小聊天应用。该聊天复用 `tool_call_demo.py` 中的路由与工具，支持：

- 学习路径建议（prompt/memory/tools 等主题）
- 系统信息概览（CPU、内存、磁盘、日期）
- 打开摄像头应用（按系统尝试调用）

可以直接在下方输入问题，例如：

- “LangChain 的 tool 模块应该怎么学？”
- “帮我查看电脑 CPU、内存和今天的日期。”
- “帮我打开摄像头窗口。”

提示：

- 未配置 `DASHSCOPE_API_KEY` 时，模型生成质量会受影响，但工具调用仍可使用。
- 打开摄像头会在本机启动相机应用，请注意隐私与系统授权。

参考：如果需要更多示例，请查看仓库的 README（RAG Demo、交互式 CLI、LangGraph 等）。
