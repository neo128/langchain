"""LangGraph 入门示例：根据问题自动路由并生成学习建议。"""

from __future__ import annotations

import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

try:  # 优先支持包内导入，兼容直接运行脚本
    from .qwen_utils import build_qwen_chat  # type: ignore
except ImportError:  # pragma: no cover
    from qwen_utils import build_qwen_chat


class WorkflowState(TypedDict, total=False):
    """在图中传递的状态。"""

    question: str
    topic: Literal["prompt", "memory", "tools", "system", "general"]
    reasoning: str
    answer: str


def classify_topic(state: WorkflowState) -> dict:
    """基于关键词的轻量分类逻辑。"""
    text = state["question"].lower()
    if any(k in text for k in ["prompt", "few-shot", "提示"]):
        topic = "prompt"
    elif any(k in text for k in ["memory", "记忆", "conversation"]):
        topic = "memory"
    elif any(k in text for k in ["tool", "工具", "agent"]):
        topic = "tools"
    elif any(k in text for k in ["cpu", "内存", "硬盘", "日期", "系统"]):
        topic = "system"
    else:
        topic = "general"

    reasoning = (
        "根据关键词判断为 Prompt 相关" if topic == "prompt"
        else "识别到 Memory 相关关键词" if topic == "memory"
        else "检测到工具 / Agent 主题" if topic == "tools"
        else "包含系统信息查询关键词" if topic == "system"
        else "未匹配特定主题，使用通用回答"
    )
    return {"topic": topic, "reasoning": reasoning}


def build_topic_chains():
    llm = build_qwen_chat(model="qwen3-coder-plus", temperature=0.2)
    parser = StrOutputParser()

    prompt_prompts = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是 LangChain Prompt 教练，请为学习者列出 3 步 Prompt 学习计划。",
            ),
            (
                "human",
                "问题: {question}\n请结合需求，输出条理清晰的计划。",
            ),
        ]
    )

    prompt_memory = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是 LangChain 记忆模块导师，请解释 memory 的用途并给出实践建议。",
            ),
            (
                "human",
                "问题: {question}\n请用要点回答。",
            ),
        ]
    )

    prompt_tools = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是 LangChain 工具与 Agent 导师，请说明如何入门 tool/agent 功能。",
            ),
            (
                "human",
                "问题: {question}\n请包含一个实际项目示例。",
            ),
        ]
    )

    prompt_system = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是系统信息助手，请提醒用户如何安全地检查本机配置并保护隐私。",
            ),
            (
                "human",
                "用户的问题: {question}\n请生成步骤提示。",
            ),
        ]
    )

    prompt_general = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是 LangChain 学习顾问，请给出泛化的自学路线和资源推荐。",
            ),
            (
                "human",
                "问题: {question}\n请以列表形式作答。",
            ),
        ]
    )

    chains = {
        "prompt": prompt_prompts | llm | parser,
        "memory": prompt_memory | llm | parser,
        "tools": prompt_tools | llm | parser,
        "system": prompt_system | llm | parser,
        "general": prompt_general | llm | parser,
    }
    return chains


def build_graph() -> StateGraph:
    chains = build_topic_chains()
    graph = StateGraph(state_schema=WorkflowState)

    graph.add_node("classify", classify_topic)

    def build_executor(topic: str):
        def _inner(state: WorkflowState) -> dict:
            answer = chains[topic].invoke({"question": state["question"]})
            return {"answer": answer}

        return _inner

    for topic in ["prompt", "memory", "tools", "system", "general"]:
        graph.add_node(f"answer_{topic}", build_executor(topic))

    graph.set_entry_point("classify")

    def route(state: WorkflowState) -> str:
        return {
            "prompt": "answer_prompt",
            "memory": "answer_memory",
            "tools": "answer_tools",
            "system": "answer_system",
            "general": "answer_general",
        }[state["topic"]]

    graph.add_conditional_edges("classify", route)

    for topic in ["prompt", "memory", "tools", "system", "general"]:
        graph.add_edge(f"answer_{topic}", END)

    return graph


def run_examples(graph) -> None:
    examples = [
        "我想系统地学习 LangChain 的 prompt 写法，给我一个计划",
        "memory 组件在对话场景下怎么使用？",
        "我要做一个智能客服，tool/agent 应该怎么入门？",
        "如何安全地查看电脑的 CPU 和内存？",
        "LangChain 是什么？初学者应该怎么开始？",
    ]

    compiled = graph.compile()

    try:
        ascii_diagram = compiled.get_graph().draw_ascii()
    except Exception:
        ascii_diagram = "(可视化生成失败，可忽略)"

    print("== LangGraph 结构图 ==")
    print(ascii_diagram)

    for question in examples:
        result = compiled.invoke({"question": question})
        print("\n== 用户问题 ==")
        print(question)
        print("-- Topic --", result.get("topic"))
        print("-- Reasoning --", result.get("reasoning"))
        print("-- Answer --")
        print(result.get("answer", ""))


def main() -> None:
    load_dotenv()
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先在 .env 中配置 DASHSCOPE_API_KEY 后再运行示例。")

    graph = build_graph()
    run_examples(graph)


if __name__ == "__main__":
    main()
