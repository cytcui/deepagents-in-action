import os
from itertools import count

from cryptography.hazmat.backends.openssl import backend
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware import SummarizationMiddleware
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

file_backend = StateBackend()

llm = ChatOpenAI(
    model_name=os.getenv("MODEL_NAME")
)


def get_long_report() -> str:
    """返回一份模拟长报告，用于观察工具结果自动卸载。"""
    lines = [
        f"{i:04d}: Virtual filesystems help agents retrieve information on demand."
        for i in range(1, 3001)
    ]

    # 在报告中间放入一条可供后续检索的结论。
    lines[1499] = "1500: 关键结论：仅在需要时读取相关片段。"

    report = "\n".join(lines)
    print("工具内部生成的报告字符数：", len(report))
    return report


agent = create_deep_agent(
    model=llm,
    tools=[get_long_report],
    backend=file_backend,
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model=llm,
            backend=file_backend,
            trigger=("messages", 6),
            keep=("messages", 2),
        )
    ],
    system_prompt="你是学习助手，每次用一句话回答，不调用工具。",
)

config = {"configurable": {"thread_id": "chapter03-file-tools"}}
seen_message_ids: set[str] = set()

print("文件工具问答实验：输入 exit、quit 或 退出 结束。")

# count(1) 从 1 持续计数，每轮等待一次用户输入。
for turn in count(1):
    try:
        question = input(f"\n第 {turn} 轮，请输入问题：").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已结束问答。")
        break

    if question.lower() in {"exit", "quit", "退出"}:
        print("已结束问答。")
        break
    if not question:
        continue

    # 只传入本轮新消息；Checkpointer 按 thread_id 恢复历史消息和文件。
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )

    # 返回值包含历史消息，按消息 ID 避免重复打印之前的工具记录。
    for msg in result["messages"]:
        if msg.id is not None:
            if msg.id in seen_message_ids:
                continue
            seen_message_ids.add(msg.id)
        if getattr(msg, "tool_calls", None):
            print("工具调用：", msg.tool_calls)
        if msg.type == "tool":
            print("工具返回：", msg.content)

    print("助手：", result["messages"][-1].content)
    print("当前文件列表：", list(result.get("files", {})))
