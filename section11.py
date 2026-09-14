#Agent:toolsの活用

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
loader = PyPDFLoader("(sample)社内規則.pdf")
docs = loader.load()

# 型ヒントと docstring をAIが読み、使い方・使いどきを判断する
@tool
def add(a: int, b: int) -> int:
    """2つの数を足し算する"""
    return a + b

llm = ChatOpenAI(model="gpt-4o-mini", 
                 temperature=0, 
                 max_tokens=100)


#モデル・ツール・システムプロンプトを渡すだけでエージェント完成
agent = create_agent(
    llm, 
    tools=[add],
    system_prompt="あなたは計算もできる助手です。",
)

#langgrapfが裏側で動いているため、"message"の方で渡す。
result = agent.invoke({"messages": [{"role": "user", "content": "128と256を足すといくつ？"}]})
print(result["messages"][-1].content)