import os

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm = ChatOpenAI(temperature=0, model_name=os.getenv("MODEL_NAME"))

agent = create_deep_agent(
    model=llm,
    checkpointer=InMemorySaver(),

)

config = {"configurable": {"thread_id": "chapter03-demo"}}

message = [
    HumanMessage(content="""
   请调用 write_file，
   把“今日学习主题：虚拟文件系统”写入 /workspace/note.md。
   """),
]

first_result = agent.invoke({"messages": message}, config=config)

print("第一次调用回答：", first_result["messages"][-1].content)

print("----------------------------------------------------------")
print("重新发起一次调用")
# 开始一次新调用：只传入文件状态和新问题，不传入上一轮消息。
read_result = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content=(
                    "请调用 read_file 读取 /workspace/note.md，告诉我内容。"
                )
            )
        ],
    },
    config=config
)

print("最终回答：", read_result["messages"][-1].content)
