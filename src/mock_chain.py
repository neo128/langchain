"""使用 FakeListLLM 演示链路流程，无需真实模型调用。"""

from __future__ import annotations

from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.fake import FakeListLLM


def build_mock_chain() -> LLMChain:
    """构建一个返回预设回答的链，便于脱机演示。"""
    prompt = PromptTemplate.from_template(
        "你是 LangChain 助教，请用一句话解释 {concept} 的用途。"
    )
    llm = FakeListLLM(
        responses=[
            "PromptTemplate 让你在链式应用中重复使用提示词。",
            "LLMChain 负责把 Prompt 与 LLM 组合成可执行的链。",
            "Memory 组件用于保存对话上下文，提升回答连贯性。",
        ]
    )
    return LLMChain(prompt=prompt, llm=llm)


def main() -> None:
    chain = build_mock_chain()
    concepts = ["PromptTemplate", "LLMChain", "Memory"]

    print("== Mock Chain 演示 ==")
    for concept in concepts:
        response = chain.invoke({"concept": concept})
        print(f"{concept}: {response['text']}")


if __name__ == "__main__":
    main()
