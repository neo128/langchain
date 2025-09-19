"""共享的 Qwen DashScope 客户端构建工具。"""

from __future__ import annotations

import os
from typing import Any
from langchain_openai import ChatOpenAI

_DEFAULT_DASHSCOPE_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def build_qwen_chat(
    model: str = "qwen3-coder-plus",
    temperature: float = 0.3,
    **kwargs: Any,
) -> ChatOpenAI:
    """返回指向 DashScope OpenAI 兼容接口的 ChatOpenAI 实例。"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 DASHSCOPE_API_KEY，请在 .env 中配置后再运行示例。")

    api_base = os.getenv("DASHSCOPE_API_BASE", _DEFAULT_DASHSCOPE_BASE)
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=api_base,
        openai_api_key=api_key,
        openai_api_base=api_base,
        **kwargs,
    )
