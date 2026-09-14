# LangGrapf

from langgraph.graph import StateGraph, START, END
from typing import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()
loader = PyPDFLoader("(sample)社内規則.pdf")
pages = loader.load()


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=100)



# グラフ全体で共有する「状態」の形。ノード間でこの辞書が受け渡される
class State(TypedDict):
    question : str
    answer: str

# ノード = 状態を受け取り、更新したい部分だけ辞書で返す関数
def answer_node(state: State):
    res = llm.invoke(state["question"])
    return {"answer": res.content}

# ノード = 状態を受け取り、更新したい部分だけ辞書で返す関数
builder = StateGraph(State)
builder.add_node("answer", answer_node)
builder.add_edge(START, "answer")
builder.add_edge("answer", END)
graph = builder.compile()   # 実行できる形に確定

print(graph.invoke({"question": "富士山の高さは？"})["answer"])
