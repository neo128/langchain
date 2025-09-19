"""LangServe + FastAPI 演示，展示如何将链路/图谱封装为可部署服务。"""

from __future__ import annotations

from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import Runnable, RunnableLambda
from langserve import add_routes

try:
    from .langgraph_demo import build_graph, classify_topic  # type: ignore
except ImportError:  # pragma: no cover
    from langgraph_demo import build_graph, classify_topic

try:
    from .qwen_utils import build_qwen_chat  # type: ignore
except ImportError:  # pragma: no cover
    from qwen_utils import build_qwen_chat

load_dotenv()

app = FastAPI(
    title="LangChain LangServe Demo",
    description=(
        "示例展示如何通过 LangServe 将 LangChain 的链 (Chain) 与 LangGraph 工作流 (Graph) 产品化。 "
        "开放两个路由：/chains/learning-helper 与 /graphs/topic-router。"
    ),
    version="0.1.0",
)


def _build_offline_learning_helper(
    prompt: ChatPromptTemplate, error_message: str
) -> Runnable:
    """在无法访问 Qwen 模型时返回一个离线 fallback runnable。"""

    def _extract_topic(prompt_value: ChatPromptValue) -> str:
        for message in reversed(prompt_value.messages):
            if isinstance(message, HumanMessage):
                content = message.content
                if isinstance(content, str):
                    if ":" in content:
                        candidate = content.split(":", 1)[-1].strip()
                        if candidate:
                            return candidate
                    stripped = content.strip()
                    if stripped:
                        return stripped
        return "该主题"

    def _offline_helper(prompt_value: ChatPromptValue) -> AIMessage:
        topic = _extract_topic(prompt_value)
        suggestions = [
            f"1. 先在 LangChain 官方文档中查阅 {topic} 的基础概念。",
            f"2. 动手实现一个围绕 {topic} 的最小 Demo，加深理解。",
            f"3. 总结遇到的问题并在社区中寻求反馈，以快速迭代 {topic} 的学习路径。",
        ]
        content = "\n".join([
            "未检测到 DASHSCOPE_API_KEY，已切换到离线学习建议模式。",
            f"错误详情: {error_message}",
            "",
            *suggestions,
        ])
        return AIMessage(content=content)

    return prompt | RunnableLambda(_offline_helper)


def build_learning_chain() -> Runnable:
    """构建一个最小的 runnable，用于生成学习建议。"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是 LangChain 助教，请根据用户的学习主题给出三条建议。",
            ),
            ("human", "主题: {topic}"),
        ]
    )
    try:
        llm = build_qwen_chat(model="qwen3-coder-plus", temperature=0.3)
    except RuntimeError as exc:  # pragma: no cover - depends on env config
        return _build_offline_learning_helper(prompt, str(exc))
    return prompt | llm


_OFFLINE_TOPIC_RESPONSES = {
    "prompt": (
        "建议先阅读 LangChain Prompt 模块的官方指南，并尝试复现 few-shot 与输出控制示例，"
        "逐步积累可复用的提示语模板。"
    ),
    "memory": (
        "可以从 ConversationBufferMemory 等基础记忆组件入手，理解它们如何在对话中维持上下文，"
        "再根据业务需求选择更高级的记忆策略。"
    ),
    "tools": (
        "先实现一个调用搜索或计算工具的简单代理，熟悉工具描述、解析结果以及错误处理，"
        "再扩展到多工具协作的复杂场景。"
    ),
    "system": (
        "请在安全的前提下查询系统信息，例如使用内置命令行工具，并确保不要在公共环境中暴露敏感配置。"
    ),
    "general": (
        "建议从官方文档与开源示例入手，了解 LangChain 的核心概念，再选择一个小项目进行实践并加入社区讨论。"
    ),
}


def _build_offline_topic_router(error_message: str) -> Runnable:
    def _offline_router(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = (inputs or {}).get("question", "").strip()
        classification = classify_topic({"question": question or ""})
        topic = classification.get("topic", "general")
        reasoning = classification.get("reasoning", "基于关键字给出默认主题。")
        answer_template = _OFFLINE_TOPIC_RESPONSES.get(topic, _OFFLINE_TOPIC_RESPONSES["general"])
        answer = answer_template
        return {
            "topic": topic,
            "reasoning": reasoning,
            "answer": answer,
            "mode": "offline",
            "error": error_message,
            "question": question,
        }

    return RunnableLambda(_offline_router).with_types(
        input_type=dict,
        output_type=dict,
    )


def build_topic_router() -> Runnable:
    try:
        return build_graph().compile()
    except RuntimeError as exc:  # pragma: no cover - depends on env config
        return _build_offline_topic_router(str(exc))


add_routes(
    app,
    build_learning_chain(),
    path="/chains/learning-helper",
)

add_routes(
    app,
    build_topic_router(),
    path="/graphs/topic-router",
)


@app.get("/", summary="服务首页")
def read_root() -> Dict[str, Any]:
    return {
        "message": "LangServe Demo 正常运行。",
        "endpoints": [
            "/chains/learning-helper/invoke",
            "/graphs/topic-router/invoke",
        ],
    }
