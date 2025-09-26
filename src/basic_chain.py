"""基础问答链示例，展示 LangChain 最基本的 Prompt + 聊天模型组合。"""

from __future__ import annotations

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

try:  # 支持包内以及脚本直接运行两种场景
    from .qwen_utils import build_qwen_chat  # type: ignore
except ImportError:  # pragma: no cover
    from qwen_utils import build_qwen_chat


def build_chain(model: str = "qwen3-coder-plus") -> LLMChain:
    """构建一个简单的 LLMChain，演示 PromptTemplate 的使用。"""
    template = (
        "你是一名资深 AI 助教，请用要点列表解释以下主题，"
        "并给出一个实践建议。\n\n主题: {topic}\n"
    )
    prompt = PromptTemplate.from_template(template)
    llm = build_qwen_chat(model=model, temperature=0.2)
    return LLMChain(prompt=prompt, llm=llm)


def main() -> None:
    load_dotenv()
    chain = build_chain()
    topic = "LangChain 的核心组件"
    response = chain.invoke({"topic": topic})

    print("== 输入主题 ==")
    print(topic)
    print("\n== AI 助教回答 ==")
    print(response["text"].strip())


if __name__ == "__main__":
    main()
