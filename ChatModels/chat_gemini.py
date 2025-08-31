from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

model_response = model.invoke("how long does it take reach karachi from hyderabad?")

print(model_response.content)