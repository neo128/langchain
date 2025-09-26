"""Chainlit + LangChain 最小可运行 Demo。

功能：
- 使用已有的 `route_and_answer`（见 `tool_call_demo.py`）进行问答与工具调用。
- 支持 .env 加载（DashScope/LangSmith 配置）。

运行：
  chainlit run src/chainlit_app.py -w
"""

from __future__ import annotations

import asyncio

import chainlit as cl
from dotenv import load_dotenv

try:  # 支持包内与脚本直接运行
    from .tool_call_demo import route_and_answer  # type: ignore
except Exception:  # pragma: no cover
    from tool_call_demo import route_and_answer


WELCOME = (
    "👋 欢迎使用 Chainlit + LangChain Demo!\n\n"
    "你可以直接用中文提问，例如：\n"
    "- LangChain 的 tool 模块应该怎么学？\n"
    "- 请总结 prompt 相关的学习步骤。\n"
    "- 帮我查看电脑 CPU、内存和今天的日期。\n"
    "- 帮我打开摄像头窗口。\n\n"
    "提示：若未配置 DASHSCOPE_API_KEY，会影响模型回答质量（工具依然可用）。"
)


@cl.on_chat_start
async def on_chat_start():
    load_dotenv()
    await cl.Message(content=WELCOME).send()


@cl.on_message
async def on_message(message: cl.Message):
    question = (message.content or "").strip()
    if not question:
        return

    try:
        # 使用 async 上下文管理器展示一个步骤，不调用不存在的 start()/end()
        async with cl.Step(name="Route & Tools", type="run"):
            # route_and_answer 是同步函数，放到线程池避免阻塞事件循环
            answer: str = await asyncio.to_thread(route_and_answer, question)
    except Exception as exc:  # pragma: no cover - 运行期容错
        await cl.Message(content=f"执行失败：{exc}").send()
        return

    await cl.Message(content=answer.strip()).send()
