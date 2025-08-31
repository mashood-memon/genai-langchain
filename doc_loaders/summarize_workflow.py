from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Initialize the model and parser
model = ChatOpenAI(model="gpt-4.1")
parser = StrOutputParser()

# Load the document
loader = TextLoader("sample_doc.txt")
docs = loader.load()
document_text = docs[0].page_content

print("Original Document Length:", len(document_text))
print("=" * 50)

# Define prompt for brief summary
brief_summary_prompt = PromptTemplate(
    template="Provide a brief 2-3 sentence summary of the following text:\n\n{text}",
    input_variables=["text"]
)

# Create the summarization chain
summarization_chain = brief_summary_prompt | model | parser

# Run the chain
summary = summarization_chain.invoke({"text": document_text})

print("SUMMARY:")
print("-" * 20)
print(summary)
