"""
AI processing utilities for summarization and question generation
"""
from typing import List, Optional
import re
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from config import CHAT_MODEL, TEMPERATURE, MAX_CONTENT_LENGTH


class AIProcessor:
    """Handle AI operations for summarization and chat"""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=TEMPERATURE, model=CHAT_MODEL)
        self.parser = StrOutputParser()
    
    def generate_document_summary(self, content: str, doc_type: str = "document") -> str:
        """Generate comprehensive summary for documents"""
        
        # Determine the appropriate prompt based on document type
        if doc_type == "video":
            template = """
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
            
            Content: {content}
            """
        else:
            template = """
            You are an expert document analyzer and summarizer.
            
            Please provide a comprehensive summary of this document in the following format:
            
            **📄 Document Overview:**
            [One sentence describing what the document is about]
            
            **🔍 Main Topics:**
            • [Main topic 1]
            • [Main topic 2]  
            • [Main topic 3]
            • [Add more as needed]
            
            **💡 Key Insights:**
            [2-3 sentences with the most important insights and conclusions]
            
            **📊 Document Structure:**
            [Brief description of how the content is organized]
            
            **🎯 Purpose & Audience:**
            [Who this document is for and what it aims to achieve]
            
            Document Content: {content}
            """
        
        summary_prompt = PromptTemplate(
            template=template,
            input_variables=["content"]
        )
        
        # Truncate content if too long
        if len(content) > MAX_CONTENT_LENGTH:
            content = content[:MAX_CONTENT_LENGTH] + "..."
        
        summary_chain = summary_prompt | self.llm | self.parser
        
        try:
            return summary_chain.invoke({"content": content})
        except Exception as e:
            raise Exception(f"Error generating summary: {str(e)}")
    
    def generate_question_suggestions(self, content: str, doc_type: str = "document") -> List[str]:
        """Generate suggested questions based on content"""
        
        if doc_type == "video":
            template = """
            Based on this video transcript, generate 6 interesting and diverse questions that viewers might want to ask. 
            Make them specific to the content and cover different aspects of the video.
            
            Format as a simple list:
            1. [Question 1]
            2. [Question 2]
            3. [Question 3]
            4. [Question 4]
            5. [Question 5]
            6. [Question 6]
            
            Video Content: {content}
            """
        else:
            template = """
            Based on this document content, generate 6 interesting and diverse questions that readers might want to ask. 
            Make them specific to the content and cover different aspects like:
            - Main concepts and definitions
            - Practical applications
            - Key insights and conclusions
            - Specific details and examples
            - Implications and consequences
            - Comparisons and relationships
            
            Format as a simple list:
            1. [Question 1]
            2. [Question 2]
            3. [Question 3]
            4. [Question 4]
            5. [Question 5]
            6. [Question 6]
            
            Document Content: {content}
            """
        
        questions_prompt = PromptTemplate(
            template=template,
            input_variables=["content"]
        )
        
        # Truncate content if too long
        if len(content) > 10000:
            content = content[:10000] + "..."
        
        questions_chain = questions_prompt | self.llm | self.parser
        
        try:
            response = questions_chain.invoke({"content": content})
            
            # Parse questions
            questions = []
            for line in response.split('\n'):
                if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
                    question = re.sub(r'^\d+\.\s*', '', line.strip())
                    question = re.sub(r'^-\s*', '', question.strip())
                    if question:
                        questions.append(question)
            
            return questions[:6]
        except Exception as e:
            raise Exception(f"Error generating questions: {str(e)}")
    
    def create_rag_chain(self, retriever):
        """Create RAG chain for chatbot functionality"""
        
        def enhanced_retrieval_and_format(question):
            """Enhanced retrieval with multiple strategies and immediate formatting"""
            try:
                # Ensure question is a string
                if not isinstance(question, str):
                    question = str(question)
                
                docs = []
                
                # Strategy 1: Standard similarity search with the exact question
                try:
                    docs = retriever.invoke(question)
                    if docs:
                        st.info(f"✅ Retrieved {len(docs)} documents using exact question")
                except Exception as e:
                    st.warning(f"⚠️ Standard search failed: {e}")
                
                # Strategy 2: If no results, try with simpler keywords
                if not docs:
                    try:
                        # Extract important words (longer than 3 characters)
                        keywords = [word for word in question.split() if len(word) > 3]
                        if keywords:
                            simple_query = " ".join(keywords[:3])  # Use first 3 keywords
                            docs = retriever.invoke(simple_query)
                            if docs:
                                st.info(f"✅ Retrieved {len(docs)} documents using keywords: {simple_query}")
                    except Exception as e:
                        st.warning(f"⚠️ Keyword search failed: {e}")
                
                # Strategy 3: Try with individual important words
                if not docs:
                    try:
                        important_words = [w for w in question.split() if len(w) > 4]
                        for word in important_words[:2]:  # Try first 2 important words
                            docs = retriever.invoke(word)
                            if docs:
                                st.info(f"✅ Retrieved {len(docs)} documents using word: {word}")
                                break
                    except Exception as e:
                        st.warning(f"⚠️ Single word search failed: {e}")
                
                # Format the retrieved documents immediately
                if not docs:
                    st.error("❌ No documents retrieved - vector store may be empty or disconnected")
                    return "ERROR: No relevant documents found in the knowledge base. The vector store appears to be empty or there may be a connection issue."
                
                # Format documents into context
                contexts = []
                for i, doc in enumerate(docs):
                    content = doc.page_content.strip()
                    if content:
                        # Include metadata if available
                        source_info = ""
                        if hasattr(doc, 'metadata') and doc.metadata:
                            source = doc.metadata.get('source', '')
                            if source:
                                source_info = f" (Source: {source})"
                        
                        contexts.append(f"Document {i+1}{source_info}:\n{content}")
                
                if not contexts:
                    st.error("❌ Retrieved documents were empty")
                    return "ERROR: Retrieved documents contained no usable content."
                
                formatted_context = "\n\n".join(contexts)
                st.success(f"✅ Successfully formatted context from {len(contexts)} documents ({len(formatted_context)} characters)")
                
                return formatted_context
                
            except Exception as e:
                st.error(f"❌ Critical retrieval error: {e}")
                return f"ERROR: Failed to retrieve documents due to: {str(e)}"
        
        # Simplified prompt that's more forgiving
        prompt = PromptTemplate(
            template="""You are a helpful AI assistant. Answer the user's question based on the provided context.

CONTEXT FROM USER'S DOCUMENTS:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Use the context above to answer the question thoroughly
- If the context contains relevant information, provide a detailed answer
- Be specific and reference details from the context
- If the context seems insufficient but contains some relevant info, do your best to answer with what's available

ANSWER:""",
            input_variables=['context', 'question']
        )
        
        # Create a simple chain that handles both retrieval and formatting in one step
        def rag_invoke(question):
            try:
                # Get formatted context
                context = enhanced_retrieval_and_format(question)
                
                # Prepare the input for the prompt
                prompt_input = {
                    'context': context,
                    'question': question
                }
                
                # Generate response
                response = (prompt | self.llm | self.parser).invoke(prompt_input)
                return response
                
            except Exception as e:
                st.error(f"❌ RAG chain error: {e}")
                return f"I encountered an error while processing your question: {str(e)}"
        
        # Return a simple callable that mimics the LangChain chain interface
        class SimpleRAGChain:
            def invoke(self, input_data):
                # Handle both string and dict inputs
                if isinstance(input_data, dict):
                    question = input_data.get("question", "")
                elif isinstance(input_data, str):
                    question = input_data
                else:
                    question = str(input_data)
                
                return rag_invoke(question)
        
        return SimpleRAGChain()
