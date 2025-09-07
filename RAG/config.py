"""
Configuration file for the Document & Video Summarizer application
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone Configuration
PINECONE_INDEX_NAME = "document-summarizer"
PINECONE_ENVIRONMENT = "us-east-1"  # Change as per your Pinecone setup

# Model Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1"
TEMPERATURE = 0.3

# Text Processing Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CONTENT_LENGTH = 15000

# Supported File Types
SUPPORTED_FILE_TYPES = {
    'pdf': 'application/pdf',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'txt': 'text/plain',
    'md': 'text/markdown'
}

# UI Configuration
PAGE_TITLE = "📚 Document & Video Summarizer"
PAGE_ICON = "📚"
LAYOUT = "wide"
