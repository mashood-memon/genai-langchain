from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming industries.",
    "Meditation can improve mental health and focus.",
    "Paris is the capital of France.",
    "Python is a popular programming language.",
    "Renewable energy is vital for a sustainable future.",
    "Space exploration expands our understanding of the universe.",
    "Healthy eating habits contribute to longevity.",
    "Blockchain technology enables secure transactions.",
    "Climate change is a global challenge."
]

res = embedding.embed_documents(texts)

print(len(res[0]))
print(str(res))