# Langsmith
#環境変数をセットするだけ　→ .envに記載すると簡単

import os
from langsmith import traceable 

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ② コードは今まで通り。書き換え不要で自動的にトレースが残る
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=100)
chain = ChatPromptTemplate.from_template("{q}") | llm
chain.invoke({"q": "LangSmithって何？"})


# traceable: LangChainを使わない普通の関数もLangSmithの記録対象にするデコレータ
@traceable
def my_step(text: str) -> str:
    return text.upper()

result = my_step("hello world")
print(result)