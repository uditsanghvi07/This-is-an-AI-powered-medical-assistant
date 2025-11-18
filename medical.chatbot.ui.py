import os
from dotenv import load_dotenv

# ✅ Load .env file FIRST before anything else
load_dotenv()

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for hospital theme (Red, White, Blue)
st.markdown("""
    <style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
    }
    
    /* Sidebar styling - Red background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #c62828 0%, #d32f2f 100%);
    }
    
    /* ALL TEXT IN SIDEBAR - WHITE COLOR */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: white !important;
    }
    
    /* Slider labels and values - WHITE */
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSlider p,
    [data-testid="stSidebar"] .stSlider span,
    [data-testid="stSidebar"] .stSlider div {
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Slider min/max values - WHITE */
    [data-testid="stSidebar"] [data-testid="stTickBar"] p,
    [data-testid="stSidebar"] [data-testid="stTickBar"] div,
    [data-testid="stSidebar"] [data-baseweb="slider"] p,
    [data-testid="stSidebar"] [data-baseweb="slider"] span {
        color: white !important;
    }
    
    /* Metrics styling - WHITE */
    [data-testid="stSidebar"] [data-testid="stMetricLabel"],
    [data-testid="stSidebar"] [data-testid="stMetricValue"],
    [data-testid="stSidebar"] [data-testid="stMetricDelta"] {
        color: white !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: bold !important;
    }
    
    /* ALL BUTTONS - Simple Blue background with white text (like first image) */
    [data-testid="stSidebar"] .stButton>button {
        background: #1976d2 !important;
        color: white !important;
        border: 2px solid #1976d2 !important;
        font-weight: 600 !important;
        padding: 0.75rem 1rem !important;
        border-radius: 10px !important;
        text-align: center !important;
        transition: none !important;
        box-shadow: none !important;
    }
    
    [data-testid="stSidebar"] .stButton>button:hover {
        background: #1565c0 !important;
        border: 2px solid #1565c0 !important;
        transform: none !important;
        box-shadow: none !important;
    }
    
    /* Info box in sidebar - Blue background with white text */
    [data-testid="stSidebar"] .stAlert {
        background: #1976d2 !important;
        color: white !important;
        border-left: 4px solid white !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stSidebar"] .stAlert p,
    [data-testid="stSidebar"] .stAlert strong,
    [data-testid="stSidebar"] .stAlert span {
        color: white !important;
    }
    
    /* User message - Blue theme */
    .user-message {
        background: linear-gradient(135deg, #1976d2 0%, #2196F3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(25, 118, 210, 0.2);
        border-left: 5px solid #0d47a1;
    }
    
    .user-message strong {
        color: white !important;
        font-size: 1.1rem;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    .user-message p {
        color: white !important;
        font-size: 1rem;
        margin: 0.5rem 0;
        line-height: 1.6;
    }
    
    .user-message small {
        color: #e3f2fd !important;
    }
    
    /* Assistant message - White with red accent */
    .assistant-message {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(198, 40, 40, 0.15);
        border-left: 5px solid #c62828;
    }
    
    .assistant-message strong {
        color: #c62828 !important;
        font-size: 1.1rem;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    .assistant-message p {
        color: #212121 !important;
        font-size: 1rem;
        margin: 0.5rem 0;
        line-height: 1.6;
    }
    
    .assistant-message small {
        color: #757575 !important;
    }
    
    /* Source documents */
    .source-doc {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-top: 0.5rem;
        border-left: 4px solid #c62828;
    }
    
    .source-doc strong {
        color: #b71c1c !important;
        font-weight: bold;
    }
    
    .source-doc p {
        color: #212121 !important;
        margin: 0.5rem 0;
    }
    
    .source-doc small {
        color: #c62828 !important;
    }
    
    /* Title styling */
    h1 {
        color: #c62828 !important;
        text-align: center;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #1976d2 !important;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Input box */
    .stChatInputContainer {
        border-top: 3px solid #c62828;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #ffebee !important;
        color: #c62828 !important;
        font-weight: bold !important;
        border-radius: 5px;
    }
    
    /* Section headers in sidebar */
    [data-testid="stSidebar"] h3 {
        border-bottom: 2px solid white;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        color: white !important;
    }
    
    /* Divider in sidebar */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #c62828 !important;
    }
    
    /* Help text in sidebar */
    [data-testid="stSidebar"] .stTooltipIcon,
    [data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"] {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

DB_FAISS_PATH = "vectorstore/db_faiss"

# Cache resources for better performance
@st.cache_resource
def get_vectorstore():
    """Load and cache the vector store"""
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        db = FAISS.load_local(
            DB_FAISS_PATH, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        return db
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

@st.cache_resource
def get_llm():
    """Initialize and cache the LLM"""
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1024,
            api_key=os.environ.get("GROQ_API_KEY")
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

def format_docs(docs):
    """Format documents for context"""
    return "\n\n".join(doc.page_content for doc in docs)

def create_qa_chain(vectorstore, llm, temperature, max_tokens):
    """Create the QA chain with custom settings"""
    
    # Update LLM with new settings
    llm.temperature = temperature
    llm.max_tokens = max_tokens
    
    prompt = ChatPromptTemplate.from_template("""
You are a helpful medical AI assistant. Use the following context to answer the question accurately.

Guidelines:
- Answer based ONLY on the provided context
- If you don't know the answer, say "I don't have enough information to answer this question"
- Be clear, concise, and professional
- Use medical terminology appropriately
- If relevant, mention that users should consult healthcare professionals

Context:
{context}

Question: {question}

Answer:""")
    
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
    
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return qa_chain, retriever

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("<h1 style='text-align: center; margin-bottom: 0; color: white !important;'>🏥</h1>", unsafe_allow_html=True)
        st.title("⚙️ Settings")
        
        # Model settings
        st.subheader("Model Configuration")
        temperature = st.slider(
            "Temperature (Creativity)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Lower = more focused, Higher = more creative"
        )
        
        max_tokens = st.slider(
            "Max Response Length",
            min_value=256,
            max_value=2048,
            value=1024,
            step=256,
            help="Maximum length of the response"
        )
        
        st.divider()
        
        # Statistics
        st.subheader("📊 Session Stats")
        if 'messages' in st.session_state:
            total_messages = len(st.session_state.messages)
            user_messages = len([m for m in st.session_state.messages if m['role'] == 'user'])
            st.metric("Total Messages", total_messages)
            st.metric("Questions Asked", user_messages)
        else:
            st.metric("Total Messages", 0)
            st.metric("Questions Asked", 0)
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Example questions
        st.subheader("💡 Example Questions")
        example_questions = [
            "What are the symptoms of diabetes?",
            "What causes high blood pressure?",
            "What are the risk factors for cancer?",
            "How is pneumonia diagnosed?"
        ]
        
        for question in example_questions:
            if st.button(question, key=question, use_container_width=True):
                st.session_state.example_question = question
        
        st.divider()
        
        # Info - Blue background with white text
        st.info("💡 **Tip**: This chatbot uses medical documents to answer your questions. Always consult a healthcare professional for medical advice.")
    
    # Main content
    st.title("🏥 Medical AI Assistant")
    st.markdown("<p class='subtitle'>Ask me anything about medical conditions, symptoms, treatments, and more!</p>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Initialize vectorstore and LLM
    vectorstore = get_vectorstore()
    llm = get_llm()
    
    if vectorstore is None or llm is None:
        st.error("⚠️ Failed to initialize the system. Please check your configuration.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(f"""
                <div class="user-message">
                    <strong>👤 You</strong>
                    <p>{message['content']}</p>
                    <small>{message.get('timestamp', '')}</small>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="assistant-message">
                    <strong>🤖 AI Assistant</strong>
                    <p>{message['content']}</p>
                    <small>{message.get('timestamp', '')}</small>
                </div>
            """, unsafe_allow_html=True)
            
            # Show source documents if available
            if 'sources' in message and message['sources']:
                with st.expander("📚 View Source Documents"):
                    for i, doc in enumerate(message['sources'], 1):
                        st.markdown(f"""
                            <div class="source-doc">
                                <strong>Source {i}</strong>
                                <p>{doc.page_content[:300]}...</p>
                                <small>Page: {doc.metadata.get('page', 'N/A')}</small>
                            </div>
                        """, unsafe_allow_html=True)
    
    # Handle example question click
    if 'example_question' in st.session_state:
        prompt = st.session_state.example_question
        del st.session_state.example_question
    else:
        # Chat input
        prompt = st.chat_input("💬 Ask your medical question here...")
    
    if prompt:
        # Add user message
        timestamp = datetime.now().strftime("%I:%M %p")
        st.session_state.messages.append({
            'role': 'user', 
            'content': prompt,
            'timestamp': timestamp
        })
        
        # Show user message immediately
        st.markdown(f"""
            <div class="user-message">
                <strong>👤 You</strong>
                <p>{prompt}</p>
                <small>{timestamp}</small>
            </div>
        """, unsafe_allow_html=True)
        
        # Generate response
        with st.spinner("🔍 Searching medical knowledge base..."):
            try:
                # Create QA chain with current settings
                qa_chain, retriever = create_qa_chain(vectorstore, llm, temperature, max_tokens)
                
                # Get source documents
                source_docs = retriever.invoke(prompt)
                
                # Get answer
                result = qa_chain.invoke(prompt)
                
                # Add assistant message
                timestamp = datetime.now().strftime("%I:%M %p")
                st.session_state.messages.append({
                    'role': 'assistant', 
                    'content': result,
                    'sources': source_docs,
                    'timestamp': timestamp
                })
                
                # Rerun to show the new message
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Please check your GROQ_API_KEY in the .env file")

if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("GROQ_API_KEY"):
        st.error("⚠️ GROQ_API_KEY not found! Please add it to your .env file.")
        st.stop()
    
    main()