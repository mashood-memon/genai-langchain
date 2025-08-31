import streamlit as st
import re
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="YouTube Video Summarizer & Chatbot",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    
    /* Main app background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #262730;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_data
def extract_video_id(url):
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

@st.cache_data
def get_video_transcript(video_id):
    """Fetch transcript from YouTube video"""
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id, languages=["en"])
        transcript = " ".join([entry.text for entry in transcript_list])
        return transcript, None
    except TranscriptsDisabled:
        return None, "Transcripts are disabled for this video."
    except Exception as e:
        return None, f"Error fetching transcript: {str(e)}"

@st.cache_resource
def create_vector_store(transcript):
    """Create vector store from transcript"""
    # Split transcript into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.create_documents([transcript])
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store

def format_docs(retrieved_docs):
    """Format retrieved documents into context"""
    context_text = "\n\n".join(
        " ".join(doc.page_content.split()) for doc in retrieved_docs
    )
    return context_text

def generate_summary(transcript):
    """Generate video summary"""
    llm = ChatOpenAI(temperature=0.3, model="gpt-4.1")
    
    summary_prompt = PromptTemplate(
        template="""
        You are an expert at summarizing YouTube videos.
        
        Please provide a comprehensive summary of this video transcript in the following format:
        
        **🎯 Main Topic:**
        [One sentence describing what the video is about]
        
        **📋 Key Points:**
        • [Key point 1]
        • [Key point 2]
        • [Key point 3]
        • [Add more as needed]
        
        **💡 Key Takeaways:**
        [2-3 sentences with the most important insights]
        
        **⏱️ Content Structure:**
        [Brief description of how the content is organized]
        
        Transcript: {transcript}
        """,
        input_variables=["transcript"]
    )
    
    # Truncate transcript if too long
    if len(transcript) > 15000:
        transcript = transcript[:15000] + "..."
    
    summary_chain = summary_prompt | llm | StrOutputParser()
    return summary_chain.invoke({"transcript": transcript})

def generate_question_suggestions(transcript):
    """Generate suggested questions"""
    llm = ChatOpenAI(temperature=0.5, model="gpt-4.1")
    
    questions_prompt = PromptTemplate(
        template="""
        Based on this video transcript, generate 6 interesting and diverse questions that viewers might want to ask. 
        Make them specific to the content and cover different aspects of the video.
        
        Format as a simple list:
        1. [Question 1]
        2. [Question 2]
        3. [Question 3]
        4. [Question 4]
        5. [Question 5]
        6. [Question 6]
        
        Transcript: {transcript}
        """,
        input_variables=["transcript"]
    )
    
    # Truncate transcript if too long
    if len(transcript) > 10000:
        transcript = transcript[:10000] + "..."
    
    questions_chain = questions_prompt | llm | StrOutputParser()
    response = questions_chain.invoke({"transcript": transcript})
    
    # Parse questions
    questions = []
    for line in response.split('\n'):
        if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
            question = re.sub(r'^\d+\.\s*', '', line.strip())
            question = re.sub(r'^-\s*', '', question.strip())
            if question:
                questions.append(question)
    
    return questions[:6]

def create_rag_chain(vector_store):
    """Create RAG chain for chatbot"""
    retriever = vector_store.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 4}
    )
    
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant that answers questions about a YouTube video.
        Use ONLY the provided context from the video transcript to answer questions.
        If the context doesn't contain enough information, say you don't have enough information from the video to answer that question.
        
        Be conversational, helpful, and reference specific parts of the video when relevant.
        
        Context from the video:
        {context}
        
        Question: {question}
        
        Answer:
        """,
        input_variables=['context', 'question']
    )
    
    llm = ChatOpenAI(temperature=0.2, model="gpt-4.1")
    parser = StrOutputParser()
    
    # Create parallel chain
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    
    # Complete RAG chain
    rag_chain = parallel_chain | prompt | llm | parser
    
    return rag_chain

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'suggested_questions' not in st.session_state:
    st.session_state.suggested_questions = []

# Main UI
st.markdown('<h1 class="main-header">🎥 YouTube Video Summarizer & Chatbot</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. **Enter YouTube URL** in the input field
    2. **Click Process** to analyze the video
    3. **Read the summary** and suggested questions
    4. **Ask questions** about the video content
    5. **Chat** with the AI about the video
    """)
    
    st.header("🔧 Settings")
    st.info("Make sure you have your OpenAI API key in your .env file")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🎬 Video Processing")
    
    # URL input
    youtube_url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Enter a valid YouTube video URL"
    )
    
    if st.button("🚀 Process Video", type="primary"):
        if youtube_url:
            video_id = extract_video_id(youtube_url)
            
            if video_id:
                with st.spinner("🔄 Processing video... This may take a moment."):
                    # Get transcript
                    transcript, error = get_video_transcript(video_id)
                    
                    if transcript:
                        try:
                            # Generate summary
                            summary = generate_summary(transcript)
                            
                            # Generate question suggestions
                            questions = generate_question_suggestions(transcript)
                            
                            # Create vector store and RAG chain
                            vector_store = create_vector_store(transcript)
                            rag_chain = create_rag_chain(vector_store)
                            
                            # Store in session state
                            st.session_state.video_processed = True
                            st.session_state.rag_chain = rag_chain
                            st.session_state.summary = summary
                            st.session_state.suggested_questions = questions
                            st.session_state.messages = []
                            
                            st.success("✅ Video processed successfully!")
                            
                        except Exception as e:
                            st.error(f"❌ Error processing video: {str(e)}")
                    else:
                        st.error(f"❌ {error}")
            else:
                st.error("❌ Invalid YouTube URL")
        else:
            st.error("❌ Please enter a YouTube URL")
    
    # Display summary if video is processed
    if st.session_state.video_processed and hasattr(st.session_state, 'summary'):
        st.header("📄 Video Summary")
        st.markdown(f'<div class="summary-box">{st.session_state.summary}</div>', unsafe_allow_html=True)
        
        # Question suggestions
        st.header("💭 Suggested Questions")
        for i, question in enumerate(st.session_state.suggested_questions):
            if st.button(f"❓ {question}", key=f"q_{i}"):
                # Add question to chat
                st.session_state.messages.append({"role": "user", "content": question})
                
                # Get answer
                with st.spinner("🤔 Thinking..."):
                    try:
                        answer = st.session_state.rag_chain.invoke(question)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"})
                
                st.rerun()

with col2:
    st.header("💬 Chat with Video")
    
    if not st.session_state.video_processed:
        st.info("👈 Please process a video first to start chatting!")
    else:
        # Chat interface
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message user-message">🧑 **You:** {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message bot-message">🤖 **AI:** {message["content"]}</div>', unsafe_allow_html=True)
        
        # Chat input
        user_question = st.text_input(
            "Ask a question about the video:",
            placeholder="What is the main topic discussed in this video?",
            key="chat_input"
        )
        
        col_send, col_clear = st.columns([3, 1])
        
        with col_send:
            if st.button("📤 Send", type="primary") and user_question:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": user_question})
                
                # Get AI response
                with st.spinner("🤔 Thinking..."):
                    try:
                        answer = st.session_state.rag_chain.invoke(user_question)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"})
                
                st.rerun()

