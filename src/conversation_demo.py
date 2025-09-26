"""使用 ConversationChain 演示 LangChain 对话记忆，接入 Qwen DashScope。"""

from __future__ import annotations

from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

try:  # 兼容包/脚本两种运行方式
    from .qwen_utils import build_qwen_chat  # type: ignore
except ImportError:  # pragma: no cover
    from qwen_utils import build_qwen_chat


def build_chain(model: str = "qwen3-coder-plus") -> ConversationChain:
    """创建附带缓冲记忆的对话链。"""
    llm = build_qwen_chat(model=model, temperature=0.5)
    memory = ConversationBufferMemory(return_messages=True)
    return ConversationChain(llm=llm, memory=memory)


def main() -> None:
    load_dotenv()
    chain = build_chain()
    user_inputs = [
        "我想学 LangChain，需要掌握哪些前置知识？",
        "给我一个练习思路，最好包含代码。",
        "能总结一下目前对话的要点吗？",
    ]

    print("== 对话演示 ==")
    for step, text in enumerate(user_inputs, start=1):
        print(f"\n[用户 {step}] {text}")
        response = chain.invoke({"input": text})
        print(f"[助手 {step}] {response['response'].strip()}")

    print("\n== 对话记忆内容 ==")
    for message in chain.memory.chat_memory.messages:
        role = "用户" if message.type == "human" else "助手"
        print(f"{role}: {message.content}")


if __name__ == "__main__":
    main()
