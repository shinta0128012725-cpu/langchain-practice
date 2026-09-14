from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=50)

prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは優秀な{role}です。"),
    ("human", "{topic}について{n}行で説明して"),
])


messages = prompt.format_messages(role="先生", topic="機械学習", n=2)
res = llm.invoke(messages)
print(res.content)