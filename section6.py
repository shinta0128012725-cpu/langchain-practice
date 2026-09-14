#0utpuParser2:(stroutputparser)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini",
                 temperature=0,
                 max_tokens=100)

prompt = ChatPromptTemplate.from_template(
    "{topic}を初心者に向けて一言で説明して"
)

parser = StrOutputParser()


chain = prompt | llm | parser

print(chain.invoke({"topic": "API"}))

for chunk in chain.stream({"topic": "API"}):
    print(chunk, end="")