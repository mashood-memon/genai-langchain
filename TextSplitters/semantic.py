from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import SemanticChunker # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
# Initialize embedding model with OpenAI
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize SemanticMeaningBased splitter with more sensitive settings
splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="standard_deviation",  # Different approach
    breakpoint_threshold_amount=0.5  # Lower = more breaks
)

# Alternative: Regular text splitter for comparison
regular_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)

# Sample text
text = """
Natural language processing (NLP) is a field of artificial intelligence. It focuses on enabling computers to understand and generate human language. Techniques include tokenization, embeddings, and transformers.

Tourism is a major global industry that involves the movement of people to destinations outside their usual environment for leisure, business, or other purposes. It encompasses various sectors including transportation, accommodation, food services, and entertainment. Popular tourist destinations often feature natural attractions like beaches, mountains, and national parks, as well as cultural sites such as museums, historical landmarks, and festivals.

Sustainable tourism has become increasingly important as the industry recognizes its environmental and social impacts. This approach focuses on minimizing negative effects while maximizing benefits for local communities. Eco-tourism, cultural tourism, and adventure tourism are growing segments that emphasize responsible travel practices and authentic experiences.

The tourism industry significantly contributes to economic development in many countries, providing employment opportunities and generating foreign exchange revenue. Digital technology has transformed how people plan and experience travel, with online booking platforms, mobile apps, and virtual reality changing the landscape of tourism marketing and customer service
.
"""

# Split text with both methods
print("SEMANTIC CHUNKER RESULTS:")
print("=" * 50)
semantic_chunks = splitter.split_text(text)

print(f"Number of chunks: {len(semantic_chunks)}")
print("=" * 50)

for i, chunk in enumerate(semantic_chunks):
    print(f"Chunk {i+1}:")
    print(chunk.strip())
    print("-" * 30)

print("\n\nREGULAR SPLITTER RESULTS (for comparison):")
print("=" * 50)
regular_chunks = regular_splitter.split_text(text)

print(f"Number of chunks: {len(regular_chunks)}")
print("=" * 50)

for i, chunk in enumerate(regular_chunks):
    print(f"Chunk {i+1}:")
    print(chunk.strip())
    print("-" * 30)