"""LangServe + FastAPI 演示，展示如何将链路/图谱封装为可部署服务。"""

from __future__ import annotations

from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph
from langserve import add_routes

try:
    from .langgraph_demo import build_graph  # type: ignore
except ImportError:  # pragma: no cover
    from langgraph_demo import build_graph

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
    llm = build_qwen_chat(model="qwen3-coder-plus", temperature=0.3)
    return prompt | llm


def build_topic_router() -> StateGraph:
    return build_graph()


add_routes(
    app,
    build_learning_chain(),
    path="/chains/learning-helper",
)

topic_router_graph = build_topic_router().compile()
add_routes(
    app,
    topic_router_graph,
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
