"""交互式演示：通过终端输入问题，动态触发工具调用。"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from tool_call_demo import route_and_answer

_PROMPT = (
    "请输入你的问题，示例：\n"
    "- LangChain 的 tool 模块应该怎么学？\n"
    "- 请总结 prompt 相关的学习步骤。\n"
    "- 帮我查看电脑 CPU、内存和今天的日期。\n"
    "- 帮我打开摄像头窗口。\n"
    "输入 exit 或按 Ctrl+D/Ctrl+C 退出。"
)


def interactive_loop() -> None:
    load_dotenv()
    print("== LangChain Tool 调用交互演示 ==")
    print(_PROMPT)

    while True:
        try:
            question = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出交互。再见！")
            return

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("退出交互。再见！")
            return

        try:
            answer = route_and_answer(question)
        except Exception as exc:  # pragma: no cover - 互动演示容错
            print(f"助手: 执行失败，错误信息: {exc}")
            continue

        print("助手:")
        print(answer.strip())


def main(argv: list[str] | None = None) -> int:
    _ = argv  # 当前无需解析命令行参数
    interactive_loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
