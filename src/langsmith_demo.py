"""LangSmith 入门示例：展示如何跟踪 LangChain 链路并创建示例数据。"""

from __future__ import annotations

import os
from typing import Iterable

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_community.llms.fake import FakeListLLM
from langchain_core.prompts import PromptTemplate
from langsmith import Client
from langsmith.run_helpers import traceable

_DATASET_NAME = "langsmith-langchain-starter"
_PROJECT_NAME = "LangChain 入门示例项目"


_FAKE_RESPONSES = [
    "1. 阅读官方文档\n2. 动手实现示例\n3. 通过项目实践巩固",
    "1. 梳理核心概念\n2. 结合案例练习\n3. 复盘总结",
    "1. 理解提示变量\n2. 尝试 few-shot\n3. 记录最佳模式",
    "1. 掌握记忆类型\n2. 练习 ConversationBufferMemory\n3. 对比其他 Memory",
    "1. 学 Agent 思维流程\n2. 熟悉 Tool Schema\n3. 构建小型自动化助手",
    "1. 规划学习目标\n2. 制定时间表\n3. 定期回顾",
]


def build_chain() -> LLMChain:
    """构建一个使用 FakeListLLM 的简单问答链，便于本地演示。"""
    prompt = PromptTemplate.from_template(
        "你是一名 LangChain 助教，请给出 3 条学习 {topic} 的建议。"
    )
    # FakeListLLM 让我们在没有真实模型的情况下观察 LangSmith 追踪。
    fake_llm = FakeListLLM(responses=_FAKE_RESPONSES)
    return LLMChain(prompt=prompt, llm=fake_llm)


@traceable(run_type="chain", name="fake_llm_training_helper")
def generate_suggestions(topic: str) -> str:
    """执行链并返回结果，@traceable 会自动把调用记录上传 LangSmith。"""
    chain = build_chain()
    result = chain.invoke({"topic": topic})
    return result["text"].strip()


def ensure_dataset(client: Client) -> str:
    """确保 LangSmith 中存在演示用的数据集，并返回数据集 ID。"""
    try:
        dataset = client.read_dataset(dataset_name=_DATASET_NAME)
    except Exception:
        dataset = client.create_dataset(
            dataset_name=_DATASET_NAME,
            description="LangChain 入门示例：主题到学习建议的演示数据集",
        )

    examples: Iterable = client.list_examples(dataset_id=dataset.id)
    if not any(True for _ in examples):
        topics = ["PromptTemplate", "Memory", "Tool/Agent"]
        for topic in topics:
            client.create_example(
                inputs={"topic": topic},
                outputs={"ideal": generate_suggestions(topic)},
                dataset_id=dataset.id,
            )
    return dataset.id


def main() -> None:
    load_dotenv()

    api_key = os.getenv("LANGCHAIN_API_KEY")
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"

    if not api_key:
        raise RuntimeError(
            "未检测到 LANGCHAIN_API_KEY，请先在 .env 中配置 LangSmith 密钥。"
        )
    if not tracing_enabled:
        raise RuntimeError(
            "请在 .env 中设置 LANGCHAIN_TRACING_V2=true，启用 LangSmith 追踪。"
        )

    # 若用户未显式设置项目名称，默认归档到示例项目，便于在控制台定位。
    os.environ.setdefault("LANGCHAIN_PROJECT", _PROJECT_NAME)

    client = Client()

    print("== LangSmith Demo ==")
    print(f"当前项目: {os.getenv('LANGCHAIN_PROJECT')}")
    dataset_id = ensure_dataset(client)
    print(f"数据集 {_DATASET_NAME} 已就绪 (ID: {dataset_id})")

    topics = [
        "PromptTemplate",
        "Memory",
        "Tool/Agent",
    ]

    for topic in topics:
        suggestion = generate_suggestions(topic)
        print(f"\n输入主题: {topic}")
        print("生成建议:")
        print(suggestion)

    print(
        "\n执行完成。请打开 LangSmith 控制台查看运行轨迹、数据集以及项目 "
        f"'{os.getenv('LANGCHAIN_PROJECT')}' 的详细记录。"
    )


if __name__ == "__main__":
    main()
