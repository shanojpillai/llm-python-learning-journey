"""
Main application file for the LocalLLM Q&A Assistant.

This is the entry point for the Streamlit application that provides a chat interface
for interacting with locally running LLMs via Ollama, with fallback to HuggingFace models.
"""

import sys
import time
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

# Import Streamlit and other dependencies
import streamlit as st

# Import local modules
from config import settings
from utils.logger import logger
from utils.helpers import check_ollama_status, format_time
from models.llm_loader import LLMManager
from models.prompt_templates import PromptTemplate

# Initialize LLM Manager
llm_manager = LLMManager()

# Get available models
available_models = llm_manager.get_available_models()

# Set page configuration
st.set_page_config(
    page_title=settings.APP_TITLE,
    page_icon=settings.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stChatMessage {
        background-color: rgba(240, 242, 246, 0.5);
    }
    .stChatMessage[data-testid="stChatMessageContent"] {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "generation_time" not in st.session_state:
    st.session_state.generation_time = None

# Sidebar with configuration options
with st.sidebar:
    st.title("📝 Settings")
    
    # Model selection
    st.subheader("Model Selection")
    
    backend_option = st.radio(
        "Select Backend:",
        ["Ollama", "HuggingFace"],
        index=0 if llm_manager.ollama_available else 1,
        disabled=not llm_manager.ollama_available
    )
    
    if backend_option == "Ollama" and llm_manager.ollama_available:
        model_option = st.selectbox(
            "Ollama Model:",
            available_models["ollama"],
            index=0 if available_models["ollama"] else 0,
            disabled=not available_models["ollama"]
        )
        use_ollama = True
    else:
        model_option = st.selectbox(
            "HuggingFace Model:",
            available_models["huggingface"],
            index=0
        )
        use_ollama = False
    
    # Generation parameters
    st.subheader("Generation Parameters")
    
    temperature = st.slider(
        "Temperature:", 
        min_value=0.1, 
        max_value=1.0, 
        value=settings.DEFAULT_TEMPERATURE,
        step=0.1,
        help="Higher values make the output more random, lower values make it more deterministic."
    )
    
    max_length = st.slider(
        "Max Length:", 
        min_value=64, 
        max_value=2048, 
        value=settings.DEFAULT_MAX_LENGTH,
        step=64,
        help="Maximum number of tokens to generate."
    )
    
    # About section
    st.subheader("About")
    st.markdown("""
    This application uses locally running LLM models to answer questions.
    - Primary: Ollama API
    - Fallback: HuggingFace Models
    """)
    
    # Show status
    st.subheader("Status")
    ollama_status = "✅ Connected" if llm_manager.ollama_available else "❌ Not available"
    st.markdown(f"**Ollama API**: {ollama_status}")
    
    if st.session_state.generation_time:
        st.markdown(f"**Last generation time**: {st.session_state.generation_time}")
    
    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("💬 LocalLLM Q&A Assistant")
st.markdown("Ask a question and get answers from a locally running LLM.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # Format prompt with template and history
            template = PromptTemplate.qa_template(
                prompt, 
                st.session_state.messages[:-1] if len(st.session_state.messages) > 1 else None
            )
            
            # Measure generation time
            start_time = time.time()
            
            # Generate response
            if use_ollama:
                response = llm_manager.generate_with_ollama(
                    template,
                    model=model_option,
                    temperature=temperature,
                    max_tokens=max_length
                )
            else:
                response = llm_manager.generate_with_hf(
                    template,
                    model=model_option,
                    temperature=temperature,
                    max_length=max_length
                )
            
            # Calculate generation time
            end_time = time.time()
            generation_time = format_time(end_time - start_time)
            st.session_state.generation_time = generation_time
            
            # Log generation info
            logger.info(f"Generated response in {generation_time} with model {model_option}")
            
            # Display response
            message_placeholder.markdown(response)
            
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            logger.error(error_message)
            message_placeholder.markdown(f"⚠️ {error_message}")

# Footer
st.markdown("---")
st.markdown(
    "Built with Streamlit, Ollama, and HuggingFace. "
    "Running LLMs locally on CPU. "
    "<br><b>Author:</b> Shanoj",
    unsafe_allow_html=True
)