from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate


#keyとpdfの読み込み
load_dotenv()
loader = PyPDFLoader("(sample)社内規則.pdf")
pages = loader.load()

#モデル作成
llm = ChatOpenAI(model="gpt-4o-mini", 
                 temperature=0, 
                 max_tokens=100)

#RAG(下準備):分割→埋め込み→保存→retriever
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = splitter.split_documents(pages)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()

#プロンプトの作成
prompt = ChatPromptTemplate.from_messages([
    ("system", "参考情報だけを参考に答えて。わからない場合は、「わからない。」と答えて"),
    ("human", "{context}\n\n質問: {question}"),
])

parser = StrOutputParser()

chatbot = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | llm 
    | parser
)

print(chatbot.invoke("有給はいつまでに申請すればいいか"))



