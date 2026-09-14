import os
from itertools import count

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, LocalShellBackend, CompositeBackend, StateBackend, StoreBackend
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from deepagents import FilesystemPermission


load_dotenv()

llm = ChatOpenAI(
    temperature=0,
    model_name=os.getenv("MODEL_NAME")
)

files = FilesystemBackend(
    root_dir="./agent-workspace",
    virtual_mode=True,
)

shell = LocalShellBackend(
    root_dir="./agent-workspace",
    virtual_mode=True,
)

file_backend = CompositeBackend(
    default=StateBackend(),
    routes={
        "/memories/": StoreBackend(
            namespace=lambda runtime: ("cuiyt",),
        ),
    },
)

store = InMemoryStore()

agent = create_deep_agent(
    model=llm,
    backend=file_backend,
    store=store,
    checkpointer=InMemorySaver(),
    system_prompt="""
    你是文件工具实验助手。
    """,
    permissions=[
        # 具体例外放在前面：这个文件可以写。
        FilesystemPermission(
            operations=["write"],
            paths=["/memories/scratch.md"],
            mode="allow",
        ),

        # 其余 memories 文件禁止写入、编辑和删除。
        FilesystemPermission(
            operations=["write"],
            paths=["/memories/**"],
            mode="deny",
        ),
    ],
)

config = {"configurable": {"thread_id": "A"}}


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

    if question.startswith("/thread "):
        thread_id = question.split(maxsplit=1)[1]
        config["configurable"]["thread_id"] = thread_id
        print("已切换线程：", thread_id)
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

