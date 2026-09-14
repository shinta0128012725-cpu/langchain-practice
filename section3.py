from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=50)
messages = [SystemMessage("あなたは関西弁のフレンドリーな先生です。"), 
            HumanMessage("Pythonのリストってなんですか？")]


res = llm.invoke(messages)
print(res.content)