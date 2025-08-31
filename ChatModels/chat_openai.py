from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4.1") 

res = model.invoke("how long does it take reach karachi from hyderabad?")

print(res.content)