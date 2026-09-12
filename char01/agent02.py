import os

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool, tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()

# llm
llm = ChatOpenAI(model=os.getenv("MODEL_NAME"))


@tool(description="获取当前时间")
def get_current_time() -> str:
    """获取当前时间。"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

class weatherInput(BaseModel):
    city: str = Field(
        description="城市名称"
    )

weather_tool = StructuredTool.from_function(
    func=get_weather,
    name="get_weather",
    description="获取城市的天气情况",
    args_schema=weatherInput
)


TOOLS = [
    weather_tool,
    get_current_time
]

TOOL_MAP = {
    tool.name: tool
    for tool in TOOLS
}


max_steps = 10

def run_agent():
    llm_with_tool = llm.bind_tools(tools=TOOLS)

    messageTemplate = ChatPromptTemplate.from_messages([
        ("system", "你是一个智能助手,用中文回答问题"),
        ("human", "{city}, 今天天气怎么样？"),
    ])

    message = messageTemplate.format_messages(city="北京")
    response = llm_with_tool.invoke(message)

    if response.tool_calls:
        for tool in response.tool_calls:
            tool_name = tool["name"]
            tool_arges = tool["args"]
            id = tool["id"]
            if TOOL_MAP.get(tool_name):
                tool_response = TOOL_MAP[tool_name].invoke(tool_arges)
                print(tool_response)

def init_messages(quest:str):
    messages = [
        SystemMessage(content="你是一个智能助手,用中文回答问题"),
        HumanMessage(content=quest)
    ]
    return messages



def run_agent_loop():
    quest = input("请输入问题:")
    result, messages = agent_loop(quest)
    print(result)
    print("----------------------------------------------------------------")
    print("以下是模型调用过程:")
    for message in messages:
        print(message)

def agent_loop(quest:str):
    messages = init_messages(quest)
    llm_with_tool = llm.bind_tools(tools=TOOLS)

    for i in range(max_steps):
        # 调用大模型
        first_response = llm_with_tool.invoke(messages)

        # 先保存模型的工具调用消息
        messages.append(first_response)
        # 不需要调用工具了，已经拿到了最终结果
        if not first_response.tool_calls:
            return first_response.content, messages


        # 还需要调用工具
        for tool in first_response.tool_calls:
            tool_name = tool["name"]
            tool_arges = tool["args"]
            id = tool["id"]
            if TOOL_MAP.get(tool_name):
                tool_response = TOOL_MAP[tool_name].invoke(tool_arges)
                messages.append(ToolMessage(content=tool_response,tool_call_id=id))

    # 超过最大步数
    return "达到最大步数，未能完成。", messages



if __name__ == "__main__":
    run_agent_loop()