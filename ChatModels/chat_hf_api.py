from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    model = "openai/gpt-oss-120b",
    task = "text-generation",
    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.3
)

model = ChatHuggingFace(llm=llm)

# Test the model
response = model.invoke("Tell me 3 facts about the roman empire")
print(response.content)