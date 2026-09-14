import os

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
llm = ChatOpenAI(model=MODEL_NAME)

def get_current_time() -> str:
    """获取当前时间。"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_deep_agent(
    model=llm,
    tools=[get_weather, get_current_time],
    system_prompt="You are a helpful assistant.",
)

quest = input("请输入问题:")
result = agent.invoke(
    {"messages": [{"role": "user", "content": quest}]}
)

print(result["messages"][-1].content)
