"""
Document processing utilities for various file types using LangChain loaders
"""
import io
import re
import tempfile
import os
from typing import List, Tuple, Optional
import streamlit as st
from urllib.parse import urlparse, parse_qs

# LangChain document loaders
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangChainDocument
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentProcessor:
    """Handle different document types using LangChain loaders"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " "]
        )
    
    def _save_uploaded_file(self, uploaded_file) -> str:
        """Save uploaded file to temporary location and return path"""
        try:
            # Create temporary file with original extension
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                return tmp_file.name
        except Exception as e:
            raise Exception(f"Error saving uploaded file: {str(e)}")
    
    def load_pdf_document(self, uploaded_file) -> List[LangChainDocument]:
        """Load PDF using LangChain PyPDFLoader"""
        try:
            # Save uploaded file temporarily
            temp_path = self._save_uploaded_file(uploaded_file)
            
            # Use PyPDFLoader
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Add source metadata
            for doc in documents:
                doc.metadata["source"] = uploaded_file.name
                doc.metadata["file_type"] = "pdf"
            
            return documents
            
        except Exception as e:
            # Clean up on error
            if 'temp_path' in locals():
                try:
                    os.unlink(temp_path)
                except:
                    pass
            raise Exception(f"Error loading PDF: {str(e)}")
    
    def load_docx_document(self, uploaded_file) -> List[LangChainDocument]:
        """Load DOCX using LangChain Docx2txtLoader"""
        try:
            # Save uploaded file temporarily
            temp_path = self._save_uploaded_file(uploaded_file)
            
            # Use Docx2txtLoader
            loader = Docx2txtLoader(temp_path)
            documents = loader.load()
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Add source metadata
            for doc in documents:
                doc.metadata["source"] = uploaded_file.name
                doc.metadata["file_type"] = "docx"
            
            return documents
            
        except Exception as e:
            # Clean up on error
            if 'temp_path' in locals():
                try:
                    os.unlink(temp_path)
                except:
                    pass
            raise Exception(f"Error loading DOCX: {str(e)}")
    
    def load_txt_document(self, uploaded_file) -> List[LangChainDocument]:
        """Load TXT using LangChain TextLoader"""
        try:
            # Save uploaded file temporarily
            temp_path = self._save_uploaded_file(uploaded_file)
            
            # Use TextLoader with encoding detection
            try:
                loader = TextLoader(temp_path, encoding='utf-8')
                documents = loader.load()
            except UnicodeDecodeError:
                # Try with different encoding
                loader = TextLoader(temp_path, encoding='latin-1')
                documents = loader.load()
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Add source metadata
            for doc in documents:
                doc.metadata["source"] = uploaded_file.name
                doc.metadata["file_type"] = "txt"
            
            return documents
            
        except Exception as e:
            # Clean up on error
            if 'temp_path' in locals():
                try:
                    os.unlink(temp_path)
                except:
                    pass
            raise Exception(f"Error loading TXT: {str(e)}")
    
    def process_uploaded_file(self, uploaded_file) -> Tuple[List[LangChainDocument], str]:
        """Process uploaded file and return LangChain documents"""
        file_type = uploaded_file.type
        filename = uploaded_file.name
        
        try:
            if file_type == "application/pdf":
                documents = self.load_pdf_document(uploaded_file)
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                documents = self.load_docx_document(uploaded_file)
            elif file_type == "text/plain":
                documents = self.load_txt_document(uploaded_file)
            else:
                raise Exception(f"Unsupported file type: {file_type}")
            
            if not documents or not any(doc.page_content.strip() for doc in documents):
                raise Exception("No text content found in the file")
            
            return documents, filename
            
        except Exception as e:
            raise Exception(f"Error processing {filename}: {str(e)}")
    
    def split_documents_into_chunks(self, documents: List[LangChainDocument]) -> List[LangChainDocument]:
        """Split documents into chunks for vector storage"""
        try:
            # Use text splitter to split the documents
            chunks = self.text_splitter.split_documents(documents)
            
            # Ensure each chunk has proper metadata
            for i, chunk in enumerate(chunks):
                if "chunk_index" not in chunk.metadata:
                    chunk.metadata["chunk_index"] = i
                    
            return chunks
            
        except Exception as e:
            raise Exception(f"Error splitting documents: {str(e)}")
    
    def get_text_from_documents(self, documents: List[LangChainDocument]) -> str:
        """Extract plain text from documents for summarization"""
        return "\n\n".join([doc.page_content for doc in documents])


class YouTubeProcessor:
    """Handle YouTube video processing"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " "]
        )
    
    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([^&\n?#]+)',
            r'youtube\.com/embed/([^&\n?#]+)',
            r'youtube\.com/v/([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    @staticmethod
    @st.cache_data
    def get_video_transcript(video_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Fetch transcript from YouTube video"""
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.fetch(video_id)
            transcript = " ".join([entry.text for entry in transcript_list])
            return transcript, None
        except TranscriptsDisabled:
            return None, "Transcripts are disabled for this video."
        except Exception as e:
            return None, f"Error fetching transcript: {str(e)}"
    
    def process_youtube_url(self, url: str) -> Tuple[Optional[List[LangChainDocument]], Optional[str], Optional[str]]:
        """Process YouTube URL and return LangChain documents"""
        video_id = self.extract_video_id(url)
        if not video_id:
            return None, None, "Invalid YouTube URL"
        
        transcript, error = self.get_video_transcript(video_id)
        if error:
            return None, video_id, error
        
        try:
            # Create LangChain document from transcript
            document = LangChainDocument(
                page_content=transcript,
                metadata={
                    "source": f"YouTube Video ({video_id})",
                    "video_id": video_id,
                    "url": url,
                    "file_type": "youtube_transcript"
                }
            )
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([document])
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
            
            return chunks, video_id, None
            
        except Exception as e:
            return None, video_id, f"Error processing transcript: {str(e)}"
    
    def get_text_from_transcript(self, documents: List[LangChainDocument]) -> str:
        """Extract plain text from transcript documents for summarization"""
        return "\n\n".join([doc.page_content for doc in documents])


def get_file_info(uploaded_file) -> dict:
    """Get information about uploaded file"""
    return {
        "name": uploaded_file.name,
        "type": uploaded_file.type,
        "size": uploaded_file.size,
        "size_mb": round(uploaded_file.size / (1024 * 1024), 2)
    }
