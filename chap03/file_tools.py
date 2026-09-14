import os
from itertools import count

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm = ChatOpenAI(
    temperature=0,
    model_name=os.getenv("MODEL_NAME")
)

agent = create_deep_agent(
    model=llm,
    backend=StateBackend(),
    checkpointer=InMemorySaver(),
    system_prompt="""
    你是文件工具实验助手。

根据用户本轮提问操作文件，有依赖的操作必须按顺序执行：
- 每条 AI 消息最多发起一个工具调用。
- 收到该工具的返回结果后，才能发起下一次调用。
- 编辑文件前必须先成功读取文件。
- 如果工具返回错误，停止本轮后续操作并报告实际错误。
- 不调用 task，不委派子 Agent。
- 最终报告必须依据工具返回，不能把计划当成执行结果。
""",
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
