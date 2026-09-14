# OutputParser1:(structure)

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=100)

#構造化データの場合
class Recipe(BaseModel):
    dish: str = Field(description="料理名")
    ingredients: list[str] = Field(description="材料")

structured_llm = llm.with_structured_output(Recipe)
res1 = structured_llm.invoke("卵焼きのレシピ教えて")
print(res1.dish, res1.ingredients)

