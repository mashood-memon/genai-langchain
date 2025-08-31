from langchain.text_splitter import  RecursiveCharacterTextSplitter # type: ignore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

# 1. Load text using LangChain TextLoader
loader = TextLoader("sample_text.txt", encoding='utf-8')
documents = loader.load()
text = documents[0].page_content

# 2. Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_documents = splitter.split_documents(documents)

# 3. Initialize embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 4. Initialize Chroma vector store
vector_store = Chroma.from_documents(
    documents=split_documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 5. Query with similarity scores
query = "How do neural networks help with language understanding?"

# Option A: vectorstore Retriever
# retriever = vector_store.as_retriever(search_kwargs={"k": 3})
# results = retriever.invoke(query)  
# for i, doc in enumerate(results):
#     print(f"Result {i+1}: {doc.page_content}")

# Option B: vectorstore Retriever with MMR
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
)

docs = retriever.invoke(query)
for i, doc in enumerate(docs):
    print(f"Result {i+1}: {doc.page_content}")

# Option C: If you need scores, stick with vector store directly
results = vector_store.similarity_search_with_score(query, k=3)