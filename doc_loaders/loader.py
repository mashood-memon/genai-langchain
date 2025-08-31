from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader, CSVLoader
from bs4 import BeautifulSoup
# text
loader = TextLoader("sample_doc.txt")
docs = loader.load()

# pdf
pdf_loader = PyPDFLoader("onlineadmission.iba.edu.pk_voucher_92250.pdf")
pdf_docs = pdf_loader.load()

# WebBaseLoader: Loads web pages via URL
loader = WebBaseLoader("https://github.com/campusx-official/langchain-document-loaders/")
web_docs = loader.load()

# csv loader
loader = CSVLoader("sample_data.csv")
for doc in loader.lazy_load():
    print(doc.page_content)  # Each row is loaded and processed only when needed
