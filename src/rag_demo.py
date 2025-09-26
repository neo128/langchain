"""最小 RAG 示例：使用 BM25 检索 + Qwen 生成。

特点：
- 纯内存语料，方便快速上手；
- 使用 `BM25Retriever`（无需向量数据库），默认检索前 4 条；
- 若未配置 `DASHSCOPE_API_KEY`，自动降级为离线模式：返回检索到的参考资料及答题提示。
"""

from __future__ import annotations

from operator import itemgetter
from typing import List

from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

try:  # 兼容包内/脚本两种运行方式
    from .qwen_utils import build_qwen_chat  # type: ignore
except ImportError:  # pragma: no cover
    from qwen_utils import build_qwen_chat


def _build_corpus() -> List[Document]:
    """构造一个简易语料库（示例用）。"""
    corpus: list[tuple[str, str]] = [
        (
            "RAG 基础",
            "检索增强生成（RAG）通过先检索相关文档，再结合大模型生成答案，以提升事实性与可控性。",
        ),
        (
            "检索实现",
            "在没有向量数据库时，可先用 BM25 等词法检索作为过渡方案，之后再替换为向量检索。",
        ),
        (
            "提示语设计",
            "RAG Prompt 常包含：用户问题、检索到的上下文、引用规则与语言风格要求。",
        ),
        (
            "引用策略",
            "回答时应尽量基于提供的上下文，并在结尾以条目形式给出引用来源标题。",
        ),
        (
            "LangChain 组合",
            "典型链路: question -> retriever -> format context -> prompt -> llm -> parser。",
        ),
        (
            "LangGraph 升级",
            "复杂场景可用 LangGraph 将检索、重写、重检索、生成等步骤编排为可观测图谱。",
        ),
        (
            "记忆与检索",
            "对话式 RAG 可混合短期对话记忆与长期知识库检索，二者职责不同。",
        ),
        (
            "工具调用",
            "遇到缺失数据或需要结构化查询时，可在 RAG 之外调用外部工具或数据库。",
        ),
    ]
    return [
        Document(page_content=content, metadata={"title": title, "source": f"demo:{i}"})
        for i, (title, content) in enumerate(corpus, start=1)
    ]


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{i+1}] {d.metadata.get('title', 'Untitled')}\n{d.page_content}" for i, d in enumerate(docs)
    )


def _build_offline_chain(retriever: BM25Retriever) -> Runnable:
    """当未配置模型时的降级链：返回检索结果 + 引导提示。"""

    def _answer(inputs: dict) -> str:
        question = (inputs or {}).get("question", "").strip()
        docs = retriever.get_relevant_documents(question) if question else []
        context = _format_docs(docs)
        parts = [
            "未检测到 DASHSCOPE_API_KEY，已切换到离线 RAG 演示模式。",
            "请先在 .env 中设置 DASHSCOPE_API_KEY 后再体验真实 LLM 生成。",
        ]
        if question:
            parts.append(f"\n== 问题 ==\n{question}")
        if context:
            parts.append(f"\n== 检索到的资料 ==\n{context}")
        else:
            parts.append("\n未检索到相关资料，换个说法再试试？")
        parts.append("\n回答建议：基于以上资料自行组织答案，并在末尾给出引用条目编号。")
        return "\n".join(parts)

    # 统一输入/输出结构：{question: str} -> str
    return RunnableLambda(_answer)


def build_rag_chain(k: int = 4) -> Runnable:
    """构建最小 RAG 链（BM25 检索 + Qwen 生成，带离线降级）。"""
    docs = _build_corpus()
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = max(1, int(k))

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是严谨的知识库问答助手。仅依据给定上下文回答问题，"
                "若上下文不足以作答，请坦诚说明并提出下一步检索建议。"
            ),
            (
                "human",
                "问题: {question}\n\n"
                "可用上下文(按相关度排序):\n{context}\n\n"
                "请给出中文回答，并在末尾列出使用到的条目编号。",
            ),
        ]
    )

    try:
        llm = build_qwen_chat(model="qwen3-coder-plus", temperature=0.1)
    except RuntimeError:
        return _build_offline_chain(retriever)

    chain: Runnable = (
        {
            "context": itemgetter("question") | retriever | RunnableLambda(_format_docs),
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def answer_question(question: str) -> str:
    chain = build_rag_chain()
    return chain.invoke({"question": question})


def main() -> None:
    load_dotenv()
    examples = [
        "RAG 的核心流程是什么？",
        "没有向量数据库时如何先做一个检索问答？",
    ]
    chain = build_rag_chain()
    for q in examples:
        print("\n== 问题 ==")
        print(q)
        ans = chain.invoke({"question": q})
        print("== 回答 ==")
        print(ans.strip())


if __name__ == "__main__":
    main()

