"""基于 LangChain 的终端控制 Demo（危险，默认关闭执行）。

本示例演示如何让模型通过 Tool 触发终端命令，支持：
- pip 安装（`pip_install`）
- 文件复制（`copy_path`，等价于 Linux `cp`）
- 文件移动（`move_path`，等价于 Linux `mv`）
- 通用 Shell 命令（`run_shell`，等价于在 bash 中执行）

安全提示：
- 默认不执行任何命令。需在 .env 中显式设置 `ALLOW_SHELL=1` 才会真正执行。
- 仍内置了一个非常保守的黑名单（如 `rm -rf`、`sudo` 等），用于兜底拦截高危命令。
- 演示用途为主，请在受控环境（如容器、临时目录）中试验，风险自负。
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field

try:  # 兼容包内/脚本直接运行两种方式
    from .qwen_utils import build_qwen_chat  # type: ignore
except Exception:  # pragma: no cover
    from qwen_utils import build_qwen_chat


# --------------------------
# 安全与运行时工具函数
# --------------------------

_DANGEROUS_PATTERNS = (
    " rm -rf ",
    " rm -r ",
    " rm -f ",
    " sudo ",
    " shutdown ",
    " reboot ",
    " halt ",
    " mkfs",
    " :(){ ",  # fork bomb
    " dd if=",
    ">/dev/",
)


def _env_allow_execute() -> bool:
    return os.getenv("ALLOW_SHELL") in {"1", "true", "True", "yes", "on"}


def _looks_dangerous(cmd: str) -> Optional[str]:
    text = f" {cmd.strip()} ".lower()
    for pat in _DANGEROUS_PATTERNS:
        if pat.strip().lower() in text:
            return pat.strip()
    return None


def _run_bash(command: str, timeout: int = 60) -> tuple[int, str]:
    """在 bash 中执行命令，返回 (returncode, combined_output)。"""
    if not _env_allow_execute():
        return (
            0,
            "[DRY-RUN] 由于未设置 ALLOW_SHELL=1，本次不执行，只展示将要运行的命令:\n"
            f"$ {command}",
        )

    blocked = _looks_dangerous(command)
    if blocked:
        return (
            1,
            f"[BLOCKED] 命令包含高危片段：{blocked}\n如确需执行，请先在受控环境中验证并移除高危参数。",
        )

    shell_path = os.getenv("SHELL", "/bin/bash")
    proc = subprocess.run(
        [shell_path, "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout


# --------------------------
# 工具定义（Tools）
# --------------------------


class PipArgs(BaseModel):
    package: str = Field(..., description="要安装的包名，例如 requests 或 requests==2.32.3")
    extra_args: Optional[str] = Field(
        None, description="可选的 pip 额外参数，例如 -U -i https://pypi.tuna.tsinghua.edu.cn/simple"
    )


@tool("pip_install", args_schema=PipArgs)
def pip_install_tool(package: str, extra_args: Optional[str] = None) -> str:
    """安装或升级一个 Python 包（使用 sys.executable -m pip）。"""
    base_cmd = f"{shlex.quote(sys.executable)} -m pip install {shlex.quote(package)}"
    if extra_args:
        base_cmd += f" {extra_args}"
    code, out = _run_bash(base_cmd, timeout=300)
    return f"$ {base_cmd}\n\n{out.strip()}\n\n(exit={code})"


class CopyArgs(BaseModel):
    src: str = Field(..., description="源路径")
    dst: str = Field(..., description="目标路径")
    recursive: bool = Field(
        False, description="若源是目录，是否递归复制（等价 cp -r）"
    )


@tool("copy_path", args_schema=CopyArgs)
def copy_path_tool(src: str, dst: str, recursive: bool = False) -> str:
    """复制文件或目录，类似 Linux 的 cp（目录需 recursive=True）。"""
    if not _env_allow_execute():
        cmd = f"cp {'-r ' if recursive else ''}{shlex.quote(src)} {shlex.quote(dst)}"
        return f"[DRY-RUN] 未开启执行，将会运行：\n$ {cmd}"

    if os.path.isdir(src):
        if not recursive:
            return "源是目录，请设置 recursive=True（等价 cp -r）。"
        if os.path.exists(dst):
            return "目标已存在，请先移除或更换路径。"
        shutil.copytree(src, dst)
        return f"已复制目录：{src} -> {dst}"
    else:
        os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
        shutil.copy2(src, dst)
        return f"已复制文件：{src} -> {dst}"


class MoveArgs(BaseModel):
    src: str = Field(..., description="源路径")
    dst: str = Field(..., description="目标路径")


@tool("move_path", args_schema=MoveArgs)
def move_path_tool(src: str, dst: str) -> str:
    """移动文件或目录，等价于 Linux 的 mv。"""
    if not _env_allow_execute():
        cmd = f"mv {shlex.quote(src)} {shlex.quote(dst)}"
        return f"[DRY-RUN] 未开启执行，将会运行：\n$ {cmd}"
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    shutil.move(src, dst)
    return f"已移动：{src} -> {dst}"


class ShellArgs(BaseModel):
    command: str = Field(..., description="需要在 bash 中执行的完整命令")


@tool("run_shell", args_schema=ShellArgs)
def run_shell_tool(command: str) -> str:
    """在 bash 中执行任意命令（高危，受 ALLOW_SHELL 与黑名单保护）。"""
    code, out = _run_bash(command)
    return f"$ {command}\n\n{out.strip()}\n\n(exit={code})"


# --------------------------
# Agent 构建与演示
# --------------------------


def build_shell_agent(verbose: bool = True) -> AgentExecutor:
    llm = build_qwen_chat(model="qwen3-coder-plus", temperature=0)
    tools = [pip_install_tool, copy_path_tool, move_path_tool, run_shell_tool]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个资深 DevOps 助手，可以使用工具在本机执行命令。\n"
                "- 在执行任何命令前，先简要说明你的计划。\n"
                "- 优先使用专用工具（pip_install/copy_path/move_path），只有当它们不满足需求时再使用 run_shell。\n"
                "- 禁止执行破坏性或提升权限的命令（如 sudo、rm -rf 等）。\n"
                "- 执行后将命令与输出原样展示，并给出简洁结论。\n"
                "- 若未开启 ALLOW_SHELL=1，则只做 DRY-RUN 并打印将要执行的命令。",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=verbose)


def demo_conversations() -> None:
    agent = build_shell_agent(verbose=True)

    samples = [
        "请把 README.md 复制到 .files/readme.copy.md",
        "把 .files/readme.copy.md 移动到 .files/readme.moved.md",
        "用 pip 安装一个 requests 包",
        "执行一个 shell 命令：echo hello && python --version",
    ]

    for q in samples:
        print("\n== 用户 ==\n" + q)
        result = agent.invoke({"input": q})
        print("== 助手 ==\n" + str(result.get("output", "")))


def interactive_loop() -> None:
    print("== Shell Tool Demo（危险操作，默认 DRY-RUN）==")
    print(
        "提示：设置 .env 中的 ALLOW_SHELL=1 才会真正执行。输入 exit 退出。\n"
        "示例：\n"
        "- 用 pip 安装 requests 包\n"
        "- 把 README.md 复制到 .files/readme.copy.md\n"
        "- 把 .files/readme.copy.md 移动到 .files/readme.moved.md\n"
        "- 执行一个 shell 命令：echo hello\n"
    )
    agent = build_shell_agent(verbose=True)
    while True:
        try:
            q = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            return
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("再见！")
            return
        try:
            result = agent.invoke({"input": q})
            print("助手:\n" + str(result.get("output", "")))
        except Exception as exc:  # pragma: no cover - 互动演示容错
            print(f"[ERROR] {exc}")


def main() -> None:
    load_dotenv()
    # 若需要快速预览，可调用 demo_conversations()；这里默认进入交互模式
    interactive_loop()


if __name__ == "__main__":
    main()

