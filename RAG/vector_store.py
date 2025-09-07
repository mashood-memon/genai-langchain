import os
import time
import tempfile
from typing import List, Optional, Tuple
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, OPENAI_API_KEY

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

class VectorStoreManager:
    """Manager class for vector store operations"""
    
    def __init__(self):
        self.vector_store = None
        self.current_namespace = None
        
    def create_vector_store_from_documents(self, documents: List[Document], namespace: Optional[str] = None) -> PineconeVectorStore:
        """Create vector store from documents"""
        self.vector_store = create_vector_store_from_documents(documents)
        self.current_namespace = namespace
        return self.vector_store
        
    def get_retriever(self, k: int = 4):
        """Get retriever from current vector store"""
        if self.vector_store:
            return self.vector_store.as_retriever(search_kwargs={"k": k})
        return None
        
    def test_retrieval(self, query: str, k: int = 4):
        """Test retrieval functionality"""
        if self.vector_store:
            return test_retrieval(self.vector_store, query, k)
        return []

def get_text_splitter():
    """Create and return a text splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

def create_vector_store_from_documents(documents: List[Document]) -> PineconeVectorStore:
    """
    Create a vector store from a list of documents.
    
    Args:
        documents: List of Document objects to be stored
        
    Returns:
        PineconeVectorStore: The created vector store
    """
    try:
        print(f"Creating vector store from {len(documents)} documents...")
        
        # Create PineconeVectorStore without namespace to avoid isolation
        vector_store = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME,
            # No namespace to avoid potential issues
        )
        
        # Wait for indexing to complete - increased wait time
        print("Waiting for documents to be indexed...")
        time.sleep(15)  # Increased from 5 to 15 seconds
        
        print(f"Successfully added {len(documents)} documents to vector store")
        
        return vector_store
        
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        raise

def get_vector_store() -> PineconeVectorStore:
    """
    Get an existing vector store instance.
    
    Returns:
        PineconeVectorStore: The vector store instance
    """
    try:
        # Create PineconeVectorStore without namespace
        vector_store = PineconeVectorStore(
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME,
        )
        
        return vector_store
        
    except Exception as e:
        print(f"Error getting vector store: {str(e)}")
        raise

def test_retrieval(vector_store: PineconeVectorStore, query: str, k: int = 4) -> List[Document]:
    """
    Test retrieval from the vector store.
    
    Args:
        vector_store: The vector store to search
        query: Search query
        k: Number of results to return
        
    Returns:
        List of retrieved documents
    """
    try:
        print(f"\n🔍 Testing retrieval with query: '{query}'")
        
        # Try similarity search
        results = vector_store.similarity_search(query, k=k)
        
        print(f"✅ Found {len(results)} results")
        for i, doc in enumerate(results, 1):
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"  {i}. {content_preview}")
            
        return results
        
    except Exception as e:
        print(f"❌ Error during retrieval: {str(e)}")
        return []

def process_uploaded_file(uploaded_file, file_type: str) -> Tuple[List[Document], Optional[PineconeVectorStore]]:
    """
    Process an uploaded file and create a vector store.
    
    Args:
        uploaded_file: The uploaded file object
        file_type: Type of the file ('pdf', 'txt', 'csv')
        
    Returns:
        Tuple of (documents, vector_store)
    """
    try:
        documents = []
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            if file_type == 'pdf':
                from document_processor import DocumentProcessor
                processor = DocumentProcessor()
                documents, _ = processor.process_uploaded_file(tmp_path)
            elif file_type == 'txt':
                from document_processor import DocumentProcessor
                processor = DocumentProcessor()
                documents, _ = processor.process_uploaded_file(tmp_path)
            elif file_type == 'csv':
                from document_processor import DocumentProcessor
                processor = DocumentProcessor()
                documents, _ = processor.process_uploaded_file(tmp_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
            if documents:
                print(f"Processed {len(documents)} documents from {file_type} file")
                
                # Create vector store from documents
                vector_store = create_vector_store_from_documents(documents)
                
                return documents, vector_store
            else:
                print("No documents were extracted from the file")
                return [], None
                
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
            
    except Exception as e:
        print(f"Error processing uploaded file: {str(e)}")
        return [], None
