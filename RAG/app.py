"""
Main Streamlit application for Document & Video Summarizer
"""
import streamlit as st
import hashlib
import time
from typing import Optional

# Import custom modules
from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT, SUPPORTED_FILE_TYPES
)
from document_processor import DocumentProcessor, YouTubeProcessor, get_file_info
from vector_store import VectorStoreManager  # Use Pinecone version
from ai_processor import AIProcessor


# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .summary-box {
        background-color: #262730;
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #404040;
    }
    .question-suggestion {
        background-color: #1e3a8a;
        color: #ffffff;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        cursor: pointer;
        border: 1px solid #3b82f6;
        transition: background-color 0.3s;
    }
    .question-suggestion:hover {
        background-color: #2563eb;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        border: 1px solid #404040;
    }
    .user-message {
        background-color: #1f2937;
        color: #ffffff;
        text-align: right;
        border-left: 4px solid #10b981;
    }
    .bot-message {
        background-color: #374151;
        color: #ffffff;
        border-left: 4px solid #3b82f6;
    }
    .file-info {
        background-color: #1f2937;
        color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .stApp {
        background-color: #0e1117;
    }
    .css-1d391kg {
        background-color: #262730;
    }
    .tab-content {
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class ChatbotApp:
    """Main application class"""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.youtube_processor = YouTubeProcessor()
        self.vector_manager = VectorStoreManager()
        self.ai_processor = AIProcessor()
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'content_processed' not in st.session_state:
            st.session_state.content_processed = False
        if 'rag_chain' not in st.session_state:
            st.session_state.rag_chain = None
        if 'suggested_questions' not in st.session_state:
            st.session_state.suggested_questions = []
        if 'current_content' not in st.session_state:
            st.session_state.current_content = None
        if 'content_type' not in st.session_state:
            st.session_state.content_type = None
        if 'summary' not in st.session_state:
            st.session_state.summary = None
    
    def render_sidebar(self):
        """Render sidebar with instructions and controls"""
        with st.sidebar:
            st.header("📋 Instructions")
            st.markdown("""
            ### 📄 For Documents:
            1. **Upload** a PDF, DOCX, or TXT file
            2. **Click Process** to analyze the document
            3. **Read the summary** and suggested questions
            4. **Chat** with the AI about the document
            
            ### 🎥 For YouTube Videos:
            1. **Enter YouTube URL** in the input field
            2. **Click Process** to analyze the video
            3. **Read the summary** and suggested questions
            4. **Chat** with the AI about the video
            """)
            
            st.header("🔧 Settings")
            st.info("Make sure you have your API keys in your .env file")
            
            # Display supported file types
            st.header("📁 Supported File Types")
            for ext, mime_type in SUPPORTED_FILE_TYPES.items():
                st.write(f"• {ext.upper()}")
            
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
            
            if st.button("🔄 Reset Session"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    
    def process_document(self, uploaded_file):
        """Process uploaded document"""
        try:
            with st.spinner("🔄 Processing document... This may take a moment."):
                # Load document using LangChain loaders
                documents, filename = self.doc_processor.process_uploaded_file(uploaded_file)
                
                # Split documents into chunks
                chunks = self.doc_processor.split_documents_into_chunks(documents)
                
                # Extract text for summarization
                text = self.doc_processor.get_text_from_documents(documents)
                
                # Create vector store
                content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
                vector_store = self.vector_manager.create_vector_store_from_documents(
                    chunks, f"doc_{content_hash}"
                )
                
                # Generate summary and questions
                summary = self.ai_processor.generate_document_summary(text, "document")
                questions = self.ai_processor.generate_question_suggestions(text, "document")
                
                # Create RAG chain with slight delay to ensure vector store is ready
                st.info("🔗 Setting up chat functionality...")
                time.sleep(1)  # Give vector store a moment to be fully ready
                retriever = self.vector_manager.get_retriever()
                rag_chain = self.ai_processor.create_rag_chain(retriever)
                
                # Update session state
                st.session_state.content_processed = True
                st.session_state.rag_chain = rag_chain
                st.session_state.summary = summary
                st.session_state.suggested_questions = questions
                st.session_state.current_content = filename
                st.session_state.content_type = "document"
                st.session_state.messages = []
                
                st.success("✅ Document processed successfully!")
                
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")
    
    def process_youtube_video(self, url):
        """Process YouTube video"""
        try:
            with st.spinner("🔄 Processing video... This may take a moment."):
                # Get transcript documents
                documents, video_id, error = self.youtube_processor.process_youtube_url(url)
                
                if error or not documents:
                    st.error(f"❌ {error if error else 'Failed to process video'}")
                    return
                
                # Extract text for summarization
                text = self.youtube_processor.get_text_from_transcript(documents)
                
                # Create vector store
                content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
                vector_store = self.vector_manager.create_vector_store_from_documents(
                    documents, f"video_{content_hash}"
                )
                
                # Generate summary and questions
                summary = self.ai_processor.generate_document_summary(text, "video")
                questions = self.ai_processor.generate_question_suggestions(text, "video")
                
                # Create RAG chain with slight delay to ensure vector store is ready
                st.info("🔗 Setting up chat functionality...")
                time.sleep(1)  # Give vector store a moment to be fully ready
                retriever = self.vector_manager.get_retriever()
                rag_chain = self.ai_processor.create_rag_chain(retriever)
                
                # Update session state
                st.session_state.content_processed = True
                st.session_state.rag_chain = rag_chain
                st.session_state.summary = summary
                st.session_state.suggested_questions = questions
                st.session_state.current_content = f"YouTube Video ({video_id})"
                st.session_state.content_type = "video"
                st.session_state.messages = []
                
                st.success("✅ Video processed successfully!")
                
        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
    
    def render_document_tab(self):
        """Render document processing tab"""
        st.header("📄 Document Processing")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your document",
            type=list(SUPPORTED_FILE_TYPES.keys()),
            help="Supported formats: PDF, DOCX, TXT"
        )
        
        if uploaded_file:
            # Show file info
            file_info = get_file_info(uploaded_file)
            st.markdown(f"""
            <div class="file-info">
                📁 <strong>File:</strong> {file_info['name']}<br>
                📊 <strong>Size:</strong> {file_info['size_mb']} MB<br>
                🏷️ <strong>Type:</strong> {file_info['type']}
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Process Document", type="primary"):
                self.process_document(uploaded_file)
    
    def render_youtube_tab(self):
        """Render YouTube processing tab"""
        st.header("🎥 YouTube Video Processing")
        
        # URL input
        youtube_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Enter a valid YouTube video URL"
        )
        
        if st.button("🚀 Process Video", type="primary"):
            if youtube_url:
                self.process_youtube_video(youtube_url)
            else:
                st.error("❌ Please enter a YouTube URL")
    
    def render_summary_and_questions(self):
        """Render summary and suggested questions"""
        if st.session_state.content_processed and st.session_state.summary:
            st.header("📄 Summary")
            st.markdown(f'<div class="summary-box">{st.session_state.summary}</div>', 
                       unsafe_allow_html=True)
            
            # Question suggestions
            st.header("💭 Suggested Questions")
            for i, question in enumerate(st.session_state.suggested_questions):
                if st.button(f"❓ {question}", key=f"q_{i}"):
                    self.ask_question(question)
    
    def render_chat_interface(self):
        """Render chat interface"""
        st.header("💬 Chat Interface")
        
        if not st.session_state.content_processed:
            st.info("👈 Please process a document or video first to start chatting!")
            return
        
        # Display current content info
        st.markdown(f"**Chatting about:** {st.session_state.current_content}")
        
        # Debug section (temporary)
        with st.expander("🔧 Debug Tools", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🧪 Test Vector Store"):
                    if hasattr(self, 'vector_manager') and self.vector_manager and self.vector_manager.vector_store:
                        self.vector_manager.test_retrieval("main topic")
            with col2:
                if st.button("🔄 Test RAG Chain"):
                    if st.session_state.rag_chain:
                        try:
                            test_answer = st.session_state.rag_chain.invoke({"question": "What is the main topic?"})
                            st.success(f"RAG Test Result: {test_answer[:100]}...")
                        except Exception as e:
                            st.error(f"RAG Test Failed: {e}")
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message user-message">🧑 **You:** {message["content"]}</div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message bot-message">🤖 **AI:** {message["content"]}</div>', 
                               unsafe_allow_html=True)
        
        # Chat input
        user_question = st.text_input(
            "Ask a question:",
            placeholder="What is this content about?",
            key="chat_input"
        )
        
        if st.button("📤 Send", type="primary") and user_question:
            self.ask_question(user_question)
    
    def ask_question(self, question: str):
        """Handle question asking"""
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Get AI response
        with st.spinner("🤔 Thinking..."):
            try:
                # Debug: Show that we're attempting to use the RAG chain
                if not st.session_state.rag_chain:
                    raise Exception("RAG chain not initialized")
                
                # Debug: Test direct vector store search first
                if hasattr(self, 'vector_manager') and self.vector_manager and self.vector_manager.vector_store:
                    try:
                        direct_results = self.vector_manager.vector_store.similarity_search(question, k=2)
                        if direct_results:
                            st.info(f"🔍 Found {len(direct_results)} relevant documents")
                        else:
                            st.warning("⚠️ No documents found in direct search")
                    except Exception as search_error:
                        st.warning(f"⚠️ Direct search failed: {search_error}")
                
                answer = st.session_state.rag_chain.invoke({"question": question})
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"Sorry, I encountered an error: {str(e)}"
                })
        
        st.rerun()
    
    def run(self):
        """Run the main application"""
        # Header
        st.markdown(f'<h1 class="main-header">{PAGE_TITLE}</h1>', unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        # Create tabs for different input types
        tab1, tab2 = st.tabs(["📄 Documents", "🎥 YouTube Videos"])
        
        with tab1:
            self.render_document_tab()
        
        with tab2:
            self.render_youtube_tab()
        
        # Summary and chat section
        col1, col2 = st.columns([1, 1])
        
        with col1:
            self.render_summary_and_questions()
        
        with col2:
            self.render_chat_interface()
        


if __name__ == "__main__":
    try:
        app = ChatbotApp()
        app.run()
    except Exception as e:
        st.error(f"Application failed to start: {str(e)}")
        st.info("Please check your configuration and API keys.")
