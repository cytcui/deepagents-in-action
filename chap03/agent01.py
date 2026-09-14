import os

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(temperature=0, model_name=os.getenv("MODEL_NAME"))

agent = create_deep_agent(
    model=llm,
    system_prompt=""""
    你是一个上下文管理实验助手。
当用户要求保存资料时，必须使用 write_file 写入指定路径。
写完后不要复述资料全文，只返回文件路径和一句话摘要。
""",

)

message = [
    HumanMessage(content="""
    请把以下资料写入 /workspace/context-demo.md：

- StateBackend 保存临时工作文件
- StoreBackend 适合跨会话持久化
- CompositeBackend 可以根据路径选择不同后端

保存后不要复述全文。"""),
]

result = agent.invoke({"messages": message})

for message in result["messages"]:
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        print("工具调用：", tool_calls)

print("最终回答：", result["messages"][-1].content)
print("State 中的文件：", list(result.get("files", {}).keys()))


print("----------------------------------------------------------")
print("重新发起一次调用")
# 开始一次新调用：只传入文件状态和新问题，不传入上一轮消息。
read_result = agent.invoke(
    {
        "files": result["files"],
        "messages": [
            HumanMessage(
                content=(
                    "请调用 read_file 读取 /workspace/context-demo.md，"
                    "然后原样输出文件的第二行。"
                )
            )
        ],
    }
)

for msg in read_result["messages"]:
    if getattr(msg, "tool_calls", None):
        print("工具调用：", msg.tool_calls)

    if msg.type == "tool":
        print("工具返回：", msg.content)

print("最终回答：", read_result["messages"][-1].content)
