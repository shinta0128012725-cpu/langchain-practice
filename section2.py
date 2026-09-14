import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


#モデルの読み込み
llm = ChatOpenAI(model = "gpt-4o-mini", temperature=0)
res = llm.invoke("こんにちは")
print(res.content)