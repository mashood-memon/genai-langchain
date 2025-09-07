# 📚 Document & Video Summarizer Chatbot

A powerful AI-powered application that can summarize and chat about both documents (PDF, DOCX, TXT) and YouTube videos using advanced RAG (Retrieval-Augmented Generation) technology.

## ✨ Features

### 📄 Document Processing
- **Multiple Format Support**: PDF, DOCX, TXT files
- **Smart Text Extraction**: Advanced parsing for different document types
- **Intelligent Chunking**: Optimal text splitting for better comprehension

### 🎥 YouTube Integration
- **Transcript Extraction**: Automatic video transcript fetching
- **URL Flexibility**: Support for various YouTube URL formats
- **Error Handling**: Graceful handling of transcript-disabled videos

### 🤖 AI-Powered Analysis
- **Comprehensive Summaries**: Structured, detailed content summaries
- **Smart Question Generation**: AI-generated relevant questions
- **Interactive Chat**: RAG-based conversational AI about your content

### 🚀 Advanced Technology
- **Pinecone Vector Database**: Scalable, cloud-based vector storage
- **OpenAI Embeddings**: High-quality text embeddings for similarity search
- **Modular Architecture**: Clean, maintainable code structure
- **Professional UI**: Modern Streamlit interface with dark theme

## 🏗️ Project Structure

```
RAG/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration and settings
├── document_processor.py     # Document parsing and processing
├── ai_processor.py          # AI operations (summarization, chat)
├── vector_store.py          # Pinecone vector database management
├── requirements.txt         # Python dependencies
├── yt_bot.ipynb            # Jupyter notebook for experimentation
└── yt_summarizer_chatbot.py # Legacy single-file version
```

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd GenAI/RAG
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the RAG directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

### 5. Configure Pinecone
Update `config.py` with your Pinecone settings:
```python
PINECONE_INDEX_NAME = "your-index-name"
PINECONE_ENVIRONMENT = "your-environment"  # e.g., "us-east-1"
```

## 🚀 Usage

### Start the Application
```bash
streamlit run app.py
```

### Using the Interface

#### 📄 For Documents:
1. **Upload**: Choose a PDF, DOCX, or TXT file
2. **Process**: Click "Process Document" 
3. **Review**: Read the AI-generated summary
4. **Interact**: Use suggested questions or ask your own
5. **Chat**: Have a conversation about the document content

#### 🎥 For YouTube Videos:
1. **Input**: Paste a YouTube URL
2. **Process**: Click "Process Video"
3. **Review**: Read the video summary
4. **Interact**: Use suggested questions or ask your own
5. **Chat**: Discuss the video content with AI

## 🔧 Configuration

### Core Settings (`config.py`)
- **Models**: OpenAI model selection (GPT-3.5-turbo/GPT-4)
- **Chunking**: Text splitting parameters
- **Embeddings**: Embedding model configuration
- **Pinecone**: Vector database settings

### Supported File Types
- **PDF**: `application/pdf`
- **DOCX**: `application/vnd.openxmlformats-officedocument.wordprocessingml.document`
- **TXT**: `text/plain`

## 🏛️ Architecture

### Document Processing Pipeline
1. **File Upload** → **Text Extraction** → **Chunking** → **Embedding** → **Vector Storage**

### AI Processing Pipeline
1. **Content Analysis** → **Summary Generation** → **Question Generation** → **RAG Setup**

### Chat Pipeline
1. **User Question** → **Vector Search** → **Context Retrieval** → **AI Response**

## 🔍 Key Components

### DocumentProcessor
- Multi-format document parsing
- Intelligent text extraction
- Error handling and validation

### VectorStoreManager
- Pinecone integration
- Namespace management
- Vector store operations

### AIProcessor
- Content summarization
- Question generation
- RAG chain creation

### ChatbotApp
- Streamlit UI management
- Session state handling
- User interaction flow

## 📊 Performance Features

- **Caching**: Streamlit caching for improved performance
- **Lazy Loading**: Efficient resource management
- **Error Recovery**: Robust error handling
- **Scalability**: Cloud-based vector storage with Pinecone

## 🎨 UI Features

- **Dark Theme**: Professional dark mode interface
- **Responsive Design**: Works on different screen sizes
- **Tab Organization**: Separate tabs for documents and videos
- **Real-time Chat**: Interactive conversation interface
- **File Information**: Detailed file metadata display

## 🔒 Security

- **Environment Variables**: Secure API key management
- **Input Validation**: File type and content validation
- **Error Handling**: Safe error messaging without exposing internals

## 🚀 Advanced Usage

### Custom Configuration
Modify `config.py` to adjust:
- Chunk sizes and overlap
- Model selection
- Temperature settings
- Vector store parameters

### Extending File Support
Add new file types in `document_processor.py`:
```python
def extract_text_from_new_format(self, uploaded_file) -> str:
    # Implementation for new file type
    pass
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain**: For the RAG framework
- **OpenAI**: For embeddings and chat models
- **Pinecone**: For vector database services
- **Streamlit**: For the web interface
- **YouTube Transcript API**: For video transcript extraction

## 🐛 Troubleshooting

### Common Issues

**Pinecone Connection Error**:
- Verify API key and environment settings
- Check Pinecone dashboard for index status

**OpenAI API Error**:
- Ensure valid API key in `.env` file
- Check API usage and billing

**Document Processing Error**:
- Verify file format is supported
- Check file size limitations
- Ensure file is not corrupted

**YouTube Transcript Error**:
- Verify video has transcripts enabled
- Check if video is publicly available
- Try different YouTube URL formats

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration guide

---

**Made with ❤️ using Streamlit, LangChain, OpenAI, and Pinecone**
