#Memory(会話履歴)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", 
                 temperature=0,
                 max_tokens=100)

prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは親切なアシスタントです。"), 
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

chain = prompt | llm

#会話ごとに履歴をしまう箱
store = {}     # {session_id : 履歴}を貯めるだけの辞書

def get_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

#chainを履歴機能で"包む"。実行のたびに get_history を使って
#     「履歴を差し込む → 返答を履歴に追記」を自動でやってくれる

bot = RunnableWithMessageHistory(
    chain, 
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

#config で「どの会話か」を session_id で伝える。同じIDなら会話が続く
cfg = {"configurable":{"session_id": "user-1"}}
print(bot.invoke({"input": "僕の名前はケンです"}, config=cfg).content)
print(bot.invoke({"input": "僕の名前わかる？"}, config=cfg).content)
