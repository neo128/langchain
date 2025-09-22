# LangChain 入门示例项目

本仓库提供一个面向初学者的 LangChain 项目示例，帮助你快速理解核心概念并上手编写简单链式应用。内容覆盖：

- 如何安装依赖并配置本地环境
- 使用 Prompt 与 Chain 构建基础问答流程
- 通过 ConversationChain 管理对话状态
- 使用 LangGraph 编排多分支链路
- 使用 LangSmith 记录链路轨迹与数据集
- 使用函数调用 (tool calling) 让模型触发外部工具（学习路径 / 系统信息 / 摄像头）
- 交互式演示，通过终端实时输入问题触发工具
- 使用 `python-dotenv` 加载 DashScope & LangSmith 配置
- 新增 RAG 示例：BM25 检索 + Qwen 生成（含离线降级）
- 新增 Chainlit 聊天界面：最小可运行 Demo，集成工具调用问答

## 环境准备

1. 确保本地安装 Python 3.10 及以上版本。
2. (推荐) 创建并激活虚拟环境：
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
4. 复制 `.env.example` 为 `.env`，并至少填入 DashScope 与 LangSmith 所需的 Key：
   ```env
   DASHSCOPE_API_KEY=sk-...
   DASHSCOPE_API_BASE=https://dashscope-intl.aliyuncs.com/compatible-mode/v1  # 可选，保留默认即可

   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   LANGCHAIN_API_KEY=lsv2-...
   LANGCHAIN_PROJECT=langchain-starter-demo
   ```

> 如果暂时不能访问 DashScope，也可以先运行示例中的 `FakeListLLM`（无需联网），观察链路执行过程。

## 运行示例

仓库的核心示例位于 `src/` 目录，可按以下命令运行：

```bash
python src/basic_chain.py            # Prompt + LLMChain 基础问答
python src/conversation_demo.py      # 带记忆的对话链
python src/langgraph_demo.py         # LangGraph 多分支工作流演示
python src/langsmith_demo.py         # LangSmith 追踪与数据集演示（需配置 LangSmith）
python src/tool_call_demo.py         # 函数调用示例（学习路径 / 系统信息 / 摄像头）
python src/tool_call_interact.py     # 交互式函数调用示例
python src/mock_chain.py             # FakeListLLM 离线演示
python src/rag_demo.py               # RAG（BM25 检索 + Qwen 生成），无 Key 自动降级离线

# 启动 Chainlit 聊天界面（集成工具调用问答）
chainlit run src/chainlit_app.py -w
```

## 使用 Chainlit 聊天界面

本项目提供了一个最小可运行的 Chainlit + LangChain 聊天 Demo，底层复用 `tool_call_demo.py` 中的函数调用（工具：学习路径、系统信息、打开摄像头）。

启动命令：

```bash
chainlit run src/chainlit_app.py -w
```

提示：

- 首次运行会生成 `.chainlit/` 配置目录与 `chainlit.md`，可据此自定义 UI、国际化与欢迎文案。
- Chainlit 默认监听 8000 端口，如已被 `uvicorn` 占用，请改端口：
  ```bash
  chainlit run src/chainlit_app.py -w -p 8001
  ```
- 在聊天框中可以直接输入：
  - “LangChain 的 tool 模块应该怎么学？”
  - “帮我查看电脑 CPU、内存和今天的日期。”
  - “帮我打开摄像头窗口。”
- 未配置 `DASHSCOPE_API_KEY` 时，模型回答可能受影响，但工具调用依旧可用。
- 摄像头工具会尝试启动本机相机应用（按系统而异），请注意隐私与权限授权。

## 目录结构说明

```
.
├── README.md
├── requirements.txt
├── .env.example
└── src/
    ├── __init__.py
    ├── basic_chain.py
    ├── conversation_demo.py
    ├── langgraph_demo.py
    ├── langsmith_demo.py
    ├── tool_call_demo.py
    ├── tool_call_interact.py
    ├── mock_chain.py
    ├── rag_demo.py
    ├── chainlit_app.py
    └── qwen_utils.py
```

## LangSmith 使用提示

- 运行 LangSmith 相关脚本前，需在 `.env` 中设置 `LANGCHAIN_TRACING_V2=true`、`LANGCHAIN_API_KEY`、`LANGCHAIN_PROJECT` 等变量。
- 调用完成后访问 [LangSmith 控制台](https://smith.langchain.com/)，即可查看链路追踪、数据集与反馈记录。
- 推荐结合 `langsmith` CLI 或网页端的 "Datasets"、"Projects" 页面，进一步执行评估或对比不同版本链路。

## 下一步学习建议

- 阅读 [LangChain 官方文档](https://python.langchain.com/docs/introduction/) 获取更丰富的组件介绍。
- 阅读 [LangGraph 文档](https://langchain-ai.github.io/langgraph/) 了解图形化编排、检查点与并行执行等特性。
- 阅读 [LangSmith 文档](https://docs.smith.langchain.com/) 掌握评估、对比、自动化测试流程。
- 在 `tool_call_demo.py` / `tool_call_interact.py` 的基础上扩展更多工具，例如检索、代码执行、系统监控、音视频控制等能力。
- 在 `langgraph_demo.py` 中尝试新增节点或使用持久化检查点，体验更复杂的工作流管理。
- 在 `langsmith_demo.py` 中将 FakeListLLM 替换为真实模型，体验端到端的调试与评估。
- 在 `mock_chain.py` 的基础上，尝试串联加载器、向量存储等更高级的链式流程。
