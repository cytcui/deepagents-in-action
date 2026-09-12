import os
from typing import Literal

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

load_dotenv()


tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
def web_search(
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = False,
):
    """
    Run a web search for the given query.
    :param query:The search query string.
    :param max_results: Maximum number of results to return.
    :param topic: The topic category for the search.
    :param include_raw_content: Whether to include raw page content.
    :return:
    """
    return tavily_client.search(query,
                                max_results = max_results,
                                topic=topic,
                                include_raw_content = include_raw_content)




def calculate(expression: str) -> str:
    """
    计算器函数，返回str类型的结果，计算表达式
    :param expression:
    :return:
    """
    return eval(expression)


def convert_currency(amount: float, from_currency: str, to_currency: str = "CNY") -> dict:
    """Convert an amount from one currency to another.

    Args:
        amount: The amount to convert.
        from_currency: The source currency code, e.g. "USD".
        to_currency: The target currency code, defaults to "CNY".
    """
    # 这里用固定汇率做演示；真实场景可接入汇率 API
    rates = {"USD": 7.2, "CNY": 1.0, "EUR": 7.8}
    cny = amount * rates[from_currency]
    return {"amount": round(cny / rates[to_currency], 2), "currency": to_currency}


def init_llm():
    MODEL_NAME = os.getenv("MODEL_NAME")
    return ChatOpenAI(model=MODEL_NAME)


system_prompt = """
你是一位专业的研究员。
你的工作是进行深入研究，然后撰写一份完整的研究报告。

你可以使用 web_search 工具搜索互联网获取信息。
"""

agent = create_deep_agent(
    model=init_llm(),
    tools=[convert_currency, calculate, web_search],
   # system_prompt="You are a helpful assistant. response with chinese.",
    system_prompt=system_prompt,
    middleware=[TodoListMiddleware()]
)

quest = input("请输入问题:")

messages = [HumanMessage(content=quest)]
result = agent.invoke({
    "messages": messages,
})

# result = agent.invoke(
#     {"messages": [{"role": "user", "content": quest}]}
# )

for message in result["messages"]:
    print(type(message).__name__)
    print("tool_calls:", getattr(message, "tool_calls", None))
    print("content:", repr(message.content))


print(result["messages"][-1].content)
