from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

# 検索対象の知識(今回は文字列。実際はPDFやWebページを読み込む)
llm = ChatOpenAI(model="gpt-4o-mini", 
                 temperature=0, 
                 max_tokens=100)


"""
# 文章の場合
text = "当社の営業時間は平日9時〜18時です。定休日は日曜です。"
"""
#①：PDFファイルを指定して、読み込む
loader = PyPDFLoader("(sample)社内規則.pdf")
pages = loader.load()

#②：そのまま、分割ステップに繋げる（元もとdocumentsになっているから分割方法が変わる）
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(pages)



"""
# 文章の場合
text = "当社の営業時間は平日9時〜18時です。定休日は日曜です。"

# ① 分割: 長文を chunk_size 文字ずつの塊に切る(この塊が検索の単位)
spliter = RecursiveCharacterTextSplitter(chunk_size=200)
docs = spliter.create_documents([text])

"""

# ② 埋め込み&保存: 各塊をベクトル化し FAISS に格納
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = FAISS.from_documents(docs, embeddings)

# ③ retriever: 質問に意味が近い塊を取り出す検索係
retriever = db.as_retriever()

# ④ プロンプト・パーサーを用意
prompt = ChatPromptTemplate.from_template(
    "次の情報だけを根拠に答えて：\n{context}\n\n質問: {question}"
)
parser = StrOutputParser()

# ⑤ | で繋ぐ。{context}=検索結果、{question}=質問 が入る
rag = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | parser
print(rag.invoke("日曜日は営業している？"))

