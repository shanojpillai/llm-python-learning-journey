import os
import sys
import streamlit as st

# Disable PyTorch's custom module watching
os.environ['STREAMLIT_DISABLE_TORCH_MODULE_WATCHING'] = '1'

# IMPORTANT: Modify Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')

# Ensure src is in Python path
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Simplified module import
def safe_import(module_path):
    """
    Safely import a module
    
    Args:
        module_path (str): Dot-separated module path
    
    Returns:
        module or None
    """
    try:
        module = __import__(module_path, fromlist=[''])
        return module
    except ImportError as e:
        st.warning(f"Could not import {module_path}: {e}")
        return None

# Page configuration
st.set_page_config(
    page_title="NLP Text Processing",
    page_icon="🧠",
    layout="wide"
)

# Default text
DEFAULT_TEXT = "Natural language processing transforms human language into a format machines can understand."

def main():
    # Initialize session state for clear trigger
    if 'clear_text' not in st.session_state:
        st.session_state.clear_text = False

    # Author and Project Details
    st.sidebar.markdown("""
    ### About the Author
    **Shanoj**
    
    [![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=flat-square&logo=github)](https://github.com/shanojpillai)
    
    *Exploring NLP Fundamentals*
    """)

    # Main title
    st.title("🧠 NLP Text Processing Pipeline")

    # Prepare modules
    cleaner = safe_import('preprocessing.cleaner')
    tokenization = safe_import('preprocessing.tokenization')
    vocab_builder = safe_import('vocabulary.vocab_builder')

    # Determine the text to display
    display_text = "" if st.session_state.clear_text else DEFAULT_TEXT

    # Text input with clear functionality
    col1, col2 = st.columns([0.85, 0.15])
    
    with col1:
        # Create a key for the text input
        input_text = st.text_area(
            "Enter text to process:", 
            value=display_text,
            height=150
        )
    
    with col2:
        # Clear button
        clear_button = st.button("🧹 Clear")
        if clear_button:
            # Set the clear trigger
            st.session_state.clear_text = True
            # Rerun to update the page
            st.rerun()

    # Reset clear trigger if text is modified
    if input_text != display_text:
        st.session_state.clear_text = False

    # Tabs for processing
    tab1, tab2, tab3 = st.tabs([
        "Preprocessing", 
        "Tokenization", 
        "Vocabulary"
    ])

    with tab1:
        st.header("📝 Text Preprocessing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Text Cleaning")
            if cleaner:
                cleaned_text = cleaner.clean_text(input_text)
                st.write("Cleaned Text:", cleaned_text)
            else:
                st.error("Cleaning module not available")
        
        with col2:
            st.subheader("Remove Special Chars")
            if cleaner and hasattr(cleaner, 'remove_special_chars'):
                no_special_chars = cleaner.remove_special_chars(input_text)
                st.write("Text without Special Characters:", no_special_chars)
            else:
                st.error("Special char removal not available")

    with tab2:
        st.header("🔤 Tokenization")
        
        if tokenization:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Basic Tokenization")
                if hasattr(tokenization, 'basic_word_tokenize'):
                    word_tokens = tokenization.basic_word_tokenize(input_text)
                    st.write(word_tokens)
            
            with col2:
                st.subheader("Advanced Tokenization")
                if hasattr(tokenization, 'advanced_tokenize'):
                    advanced_tokens = tokenization.advanced_tokenize(input_text)
                    st.write(advanced_tokens)
            
            with col3:
                st.subheader("Character Tokenization")
                if hasattr(tokenization, 'character_tokenize'):
                    char_tokens = tokenization.character_tokenize(input_text)
                    st.write(char_tokens)
        else:
            st.error("Tokenization module not available")

    with tab3:
        st.header("📚 Vocabulary")
        
        if vocab_builder:
            col1, col2 = st.columns(2)
            
            with col1:
                max_vocab_size = st.slider("Maximum Vocabulary Size", 100, 10000, 1000)
            
            with col2:
                min_freq = st.slider("Minimum Token Frequency", 1, 10, 1)
            
            # Use advanced tokenize or fallback to split
            tokenizer = (tokenization.advanced_tokenize 
                         if tokenization and hasattr(tokenization, 'advanced_tokenize') 
                         else str.split)
            
            try:
                vocab = vocab_builder.build_vocabulary(
                    [input_text], 
                    tokenizer=tokenizer, 
                    max_size=max_vocab_size, 
                    min_freq=min_freq,
                    add_special_tokens=True
                )
                
                st.write("Vocabulary Size:", len(vocab))
                st.write("Top 20 Tokens:")
                sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
                st.write(sorted_vocab[:20])
            except Exception as e:
                st.error(f"Vocabulary building failed: {e}")
        else:
            st.error("Vocabulary module not available")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; margin-top: 20px;">
    🚀 Crafted with ❤️ by Shanoj | NLP Exploration
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()