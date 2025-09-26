"""演示手动 orchestrate 模型函数调用 (tool calling) 的流程。"""

from __future__ import annotations

import datetime as dt
import os
import platform
import shutil
import subprocess
from typing import Literal, Optional, Sequence

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

try:  # 兼容从包导入及脚本直接运行
    from .qwen_utils import build_qwen_chat  # type: ignore
except ImportError:  # pragma: no cover
    from qwen_utils import build_qwen_chat


def _run_sysctl_cpu_brand() -> str | None:
    if platform.system().lower() != "darwin":
        return None
    try:
        output = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return output.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _get_total_memory_bytes() -> int | None:
    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            if page_size > 0 and pages > 0:
                return page_size * pages
        except (ValueError, OSError):
            return None
    if platform.system().lower() == "windows":
        try:
            import ctypes  # type: ignore

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullTotalPhys)
        except Exception:  # pragma: no cover - best effort
            return None
    return None


def _format_bytes(num: int | None) -> str:
    if num is None or num <= 0:
        return "未知"
    value = float(num)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def get_learning_path(topic: str) -> str:
    catalog = {
        "prompt": "1. 学习 PromptTemplate 的变量替换\n2. 尝试 Few-shot Prompt\n3. 阅读 LangChain Prompt 管理章节",
        "memory": "1. 回顾对话式 LLM 的上下文处理\n2. 了解 ConversationBufferMemory\n3. 实验其他 Memory 变体",
        "tools": "1. 学习 Tool 与 AgentExecutor 的基本概念\n2. 实战 create_openai_functions_agent\n3. 组合检索、计算等外部能力",
    }
    key = topic.strip().lower()
    return catalog.get(
        key,
        "推荐先掌握 LangChain 的 Prompt、Chain、Memory 和 Tool 四大模块，再根据项目需求深入。",
    )


def get_system_overview() -> str:
    cpu_brand = _run_sysctl_cpu_brand() or platform.processor() or platform.machine()
    total_memory = _format_bytes(_get_total_memory_bytes())
    disk = shutil.disk_usage(os.path.abspath(os.sep))
    total_disk = _format_bytes(disk.total)
    today = dt.datetime.now().strftime("%Y-%m-%d")

    return (
        f"CPU: {cpu_brand}\n"
        f"内存总量: {total_memory}\n"
        f"系统盘容量: {total_disk}\n"
        f"今天日期: {today}"
    )


def _launch_camera_commands() -> dict[str, Sequence[str]]:
    system = platform.system().lower()
    if system == "darwin":
        return {
            "Photo Booth": ["open", "/System/Applications/Photo Booth.app"],
            "QuickTime Player": ["open", "-a", "QuickTime Player"],
        }
    if system == "windows":
        return {
            "Windows Camera (PowerShell)": [
                "powershell",
                "-Command",
                "Start-Process",
                "microsoft.windows.camera:",
            ],
            "Windows Camera (cmd)": [
                "cmd",
                "/c",
                "start",
                "microsoft.windows.camera:",
            ],
        }
    return {
        "Cheese (xdg-open)": ["xdg-open", "cheese"],
        "Cheese": ["cheese"],
    }


def open_camera_app() -> str:
    attempts = _launch_camera_commands()
    last_error: str | None = None
    for label, command in attempts.items():
        try:
            subprocess.Popen(command)
            return f"已尝试通过 {label} 启动摄像头应用。"
        except FileNotFoundError as exc:
            last_error = f"{label}: {exc}"
        except Exception as exc:  # pragma: no cover - best effort
            last_error = f"{label}: {exc}"
    return (
        "未能成功启动摄像头应用。" if last_error is None else f"未能启动摄像头，应急信息: {last_error}"
    )


class RouterDecision(BaseModel):
    """模型规划的工具调用决策。"""

    tool: Literal[
        "get_learning_path",
        "get_system_overview",
        "open_camera_app",
    ] = Field(..., description="需要调用的工具名称")
    topic: Optional[str] = Field(None, description="当 tool 为 get_learning_path 时需要的主题关键词")
    reason: str = Field(..., description="为何选择该工具的简要说明")


def build_router_chain():
    llm = build_qwen_chat(model="qwen3-coder-plus", temperature=0)
    parser = JsonOutputParser(pydantic_object=RouterDecision)
    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是 LangChain 助教，需要在不同工具之间做路由决策。"
            "仅从以下选项中选择一个工具: get_learning_path, get_system_overview, open_camera_app。"
            "输出必须是有效 JSON。",
        ),
        (
            "human",
            "问题: {question}\n\n请分析并根据需要提供的工具作出决策。",
        ),
        (
            "system",
            "格式要求:\n{format_instructions}",
        ),
    ]).partial(format_instructions=format_instructions)

    return prompt | llm | parser


def build_answer_chain():
    llm = build_qwen_chat(model="qwen3-coder-plus", temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是 LangChain 助教，根据提供的工具输出，为用户生成自然语言回复。",
        ),
        (
            "human",
            "用户提问: {question}\n"
            "工具名称: {tool}\n"
            "工具说明: {reason}\n"
            "工具输出:\n{tool_output}\n\n"
            "请用中文整理成友好且结构化的回答，若涉及摄像头操作需提醒用户注意隐私。",
        ),
    ])
    return prompt | llm | StrOutputParser()


def route_and_answer(question: str) -> str:
    router_chain = build_router_chain()
    decision_raw = router_chain.invoke({"question": question})
    decision = (
        decision_raw
        if isinstance(decision_raw, RouterDecision)
        else RouterDecision.model_validate(decision_raw)
    )

    if decision.tool == "get_learning_path":
        topic = (decision.topic or "tools").lower()
        tool_output = get_learning_path(topic)
    elif decision.tool == "get_system_overview":
        tool_output = get_system_overview()
    else:
        tool_output = open_camera_app()

    answer_chain = build_answer_chain()
    return answer_chain.invoke(
        {
            "question": question,
            "tool": decision.tool,
            "reason": decision.reason,
            "tool_output": tool_output,
        }
    )


def main() -> None:
    try:
        from .env_utils import init_env  # type: ignore
    except Exception:
        from env_utils import init_env  # type: ignore
    init_env()
    questions = [
       
        "顺便帮我看看这台电脑的 CPU 和内存情况，以及今天日期。",
        "可以帮我打开摄像头窗口吗？",
    ]

    for question in questions:
        print("\n== 用户问题 ==")
        print(question)
        response = route_and_answer(question)
        print("== 助手回答 ==")
        print(response.strip())


if __name__ == "__main__":
    main()
