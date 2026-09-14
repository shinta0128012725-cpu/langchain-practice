# RAFAS(RAGの評価)

from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# 採点役(ジャッジ)のLLM。RAGASはLLMに"採点"させて品質を数値化する
judge = LangchainLLMWrapper(ChatOpenAI(model= "gpt-4o-mini"))

# 評価用データ。1件=「質問 / RAGの回答 / 使った文脈 / 正解」の4点セット
datasets = EvaluationDataset.from_list([
    {
        "user_input": "日曜日は営業している？", 
        "response": "いいえ、日曜は定休日です。",           # RAGが出した回答
        "retrieved_contexts": ["定休日は日曜です。"],       # 検索で渡した文脈
        "reference": "日曜日は定休日で営業していない。",     # 人が用意した正解
    },
])

# 測りたい指標を選んで evaluate に渡すだけ
result = evaluate(
    dataset=datasets,
    metrics=[
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        ],
        llm=judge,
)

print(result)