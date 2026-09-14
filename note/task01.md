环境配置情况

python 版本
```shell
(.venv) cuiyt@macbook-pro deepagents-in-action % python --version
Python 3.11.3
```

deepagent 版本
```shell
(.venv) cuiyt@macbook-pro deepagents-in-action % pip show deepagents
Name: deepagents
Version: 0.7.13
Summary: Production-ready, extensible agent harness with a built-in filesystem and context management, sub-agent delegation, skills, and long-term memory.
Home-page: 
Author: 
Author-email: 
License: MIT
Location: /Users/cuiyt/PycharmProjects/deepagents-in-action/.venv/lib/python3.11/site-packages
Requires: langchain, langchain-anthropic, langchain-core, langchain-google-genai, langsmith, packaging, wcmatch
Required-by: 
(.venv) cuiyt@macbook-pro deepagents-in-action % 

```


代码
```python
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
```

smith 查询trace

https://apac.smith.langchain.com/public/5ff95473-f308-4214-ba3b-8ef8e6ed2524/r/01a09fee-9dd5-7840-bbf3-1db2bbcbb9f8?start_time=2026-09-14T12%3A40%3A10.453683&scroll_to=metadata
