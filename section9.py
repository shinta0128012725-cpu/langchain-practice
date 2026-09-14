# ２段構造のリランキング


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

#リランキング関連
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

#①：PDFの読み込み
load_dotenv()
loader = PyPDFLoader("(sample)社内規則.pdf")
pages = loader.load()

#②：chunkに分割
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
docs = splitter.split_documents(pages)

#③：ベクトルに落としてdbへ格納
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = FAISS.from_documents(docs, embeddings)

#ここからリランキング
#⑴：すでに作ってある db(ベクトルDB)から、まず広めに20件ほど拾う retriever
base_retriever = db.as_retriever(search_kwargs={"k": 20})

#⑵：リランカー本体。質問と各文書のペアを見て「関連度」を採点し直す専用モデル
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
reranker = CrossEncoderReranker(model=model, top_n=3)

#⑶：rerankerを入れた二段構えのretriverの作成
retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever
)
docs = retriever.invoke("日曜は営業してる？")
for d in docs:
    print(d.page_content)



