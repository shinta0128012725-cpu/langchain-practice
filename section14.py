# VLM:画像データの読み込み


import base64
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader


load_dotenv()
loader = PyPDFLoader("(sample)社内規則.pdf")
pages = loader.load()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=100)

# 画像は base64 文字列に変換してからLLMへ渡す
with open("(sample)_table_.jpg", "rb") as f:
    img = base64.b64encode(f.read()).decode()

# content にテキストと画像のパーツを並べると、両方をまとめて1回で送れる
# OpenAIの形式では image_url を {"url": "..."} の辞書で渡す


message = HumanMessage(content=[
    {"type": "text", "text": "この画像の内容を説明して"},
    {"type": "image_url", 
     "image_url": {"url": f"data:image/jpeg;base64,{img}"}}, 
    ])

res = llm.invoke([message])
print(res.content)