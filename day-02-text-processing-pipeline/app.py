import os
import sys
import torch
import streamlit as st
import traceback
import matplotlib.pyplot as plt

# IMPORTANT: Modify Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')

# Ensure src is in Python path
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="LLM Text Processing Pipeline",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .reportview-container {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 10px 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #e6f2ff;
        color: #0068b7;
    }
    .learning-note {
        background-color: #e6f3ff;
        border-left: 4px solid #0068b7;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Imports
try:
    # Preprocessing
    from preprocessing.cleaner import clean_text, remove_special_chars
    from preprocessing.tokenization import (
        basic_word_tokenize, 
        advanced_tokenize, 
        character_tokenize
    )
    
    # Vocabulary
    from vocabulary.vocab_builder import (
        build_vocabulary, 
        tokenize_and_encode, 
        decode_token_ids
    )
    
    # Embeddings
    from models.embeddings import (
        create_embeddings, 
        visualize_embeddings, 
        calculate_similarity
    )
    
    # Language Model
    from models.language_model import (
        SimpleLanguageModel, 
        generate_text, 
        train_model
    )
    
    # Visualization
    from visualization.visualize import plot_token_distribution
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error(traceback.format_exc())

def display_learning_note(note):
    """
    Display a styled learning note with key insights.
    
    Args:
        note (str): The learning note to display
    """
    st.markdown(f'<div class="learning-note">{note}</div>', unsafe_allow_html=True)

def main():
    # Title and introduction
    st.title("🧠 LLM Text Processing Pipeline")
    st.markdown("""
    **Explore the fundamental mechanics of how language models process text!**
    
    This interactive pipeline breaks down text processing into key components, 
    allowing you to experiment with and understand each stage of natural language processing.
    """)
    
    # Learning objective overview
    display_learning_note("""
    🔍 Learning Objective: Understand how raw text transforms into a format 
    language models can process by exploring tokenization, vocabulary building, 
    embeddings, and text generation.
    """)
    
    # Tabs for different pipeline components
    tabs = [
        "Preprocessing", 
        "Tokenization", 
        "Vocabulary", 
        "Embeddings", 
        "Language Model",
        "Distribution"
    ]
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tabs)
    
    # Shared input text area
    input_text = st.text_area(
        "Enter text to explore:", 
        "Natural language processing transforms human language into a format machines can understand.",
        help="Try different types of text to see how the pipeline handles various inputs!"
    )
    
    with tab1:
        st.header("📝 Text Preprocessing")
        display_learning_note("""
        Text preprocessing is the first step in preparing text for machine learning. 
        It involves cleaning and normalizing text to create a consistent input format.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Text Cleaning")
            if st.button("Clean Text"):
                cleaned_text = clean_text(input_text)
                st.write("Cleaned Text:", cleaned_text)
                st.info("Notice how text is converted to lowercase and stripped of extra whitespace.")
        
        with col2:
            st.subheader("Special Character Removal")
            if st.button("Remove Special Chars"):
                no_special_chars = remove_special_chars(input_text)
                st.write("Text without Special Characters:", no_special_chars)
                st.info("Punctuation and special symbols are removed, leaving only words.")
    
    with tab2:
        st.header("🔤 Tokenization Methods")
        display_learning_note("""
        Tokenization breaks text into smaller units (tokens) that models can process. 
        Different strategies dramatically affect how language is interpreted.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Basic Word Tokenization")
            word_tokens = basic_word_tokenize(input_text)
            st.write(word_tokens)
            st.info("Simple splitting on whitespace with basic punctuation handling.")
        
        with col2:
            st.subheader("Advanced Tokenization")
            advanced_tokens = advanced_tokenize(input_text)
            st.write(advanced_tokens)
            st.info("More sophisticated handling of punctuation and special characters.")
        
        with col3:
            st.subheader("Character Tokenization")
            char_tokens = character_tokenize(input_text)
            st.write(char_tokens)
            st.info("Each character becomes a separate token - useful for languages with complex word structures.")
    
    with tab3:
        st.header("📚 Vocabulary Building")
        display_learning_note("""
        Vocabulary creation maps tokens to unique IDs, allowing models to work with numerical representations. 
        Key considerations include vocabulary size and token frequency.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_vocab_size = st.slider("Maximum Vocabulary Size", 100, 10000, 1000)
        
        with col2:
            min_freq = st.slider("Minimum Token Frequency", 1, 10, 1)
        
        if st.button("Build Vocabulary"):
            vocab = build_vocabulary(
                [input_text], 
                tokenizer=advanced_tokenize, 
                max_size=max_vocab_size, 
                min_freq=min_freq,
                add_special_tokens=True
            )
            
            st.write("Vocabulary Size:", len(vocab))
            st.write("Top 20 Tokens:")
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
            st.write(sorted_vocab[:20])
            
            st.info("Special tokens like <|unk|> help handle out-of-vocabulary words.")
    
    with tab4:
        st.header("🌐 Word Embeddings")
        display_learning_note("""
        Word embeddings create vector representations that capture semantic relationships. 
        Similar words cluster together in the embedding space.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            embedding_dim = st.slider("Embedding Dimension", 10, 300, 100)
        
        with col2:
            viz_method = st.selectbox("Visualization Method", ['PCA', 't-SNE'])
        
        with col3:
            top_n_words = st.slider("Top N Words to Visualize", 10, 500, 100)
        
        if st.button("Generate and Visualize Embeddings"):
            try:
                # First build vocabulary
                vocab = build_vocabulary(
                    [input_text], 
                    tokenizer=advanced_tokenize, 
                    max_size=1000,
                    add_special_tokens=True
                )
                
                # Create embeddings
                embeddings = create_embeddings(vocab, embedding_dim=embedding_dim)
                
                # Visualize embeddings
                fig = visualize_embeddings(vocab, embeddings, method=viz_method, top_n=top_n_words)
                st.pyplot(fig)
                plt.close(fig)
                
                # Word similarity demo
                st.subheader("Word Similarity")
                tokens = list(vocab.keys())
                col1, col2 = st.columns(2)
                
                with col1:
                    word1 = st.selectbox("First Word", tokens)
                
                with col2:
                    word2 = st.selectbox("Second Word", tokens)
                
                # Calculate similarity
                if st.button("Calculate Similarity"):
                    similarity = calculate_similarity(word1, word2, vocab, embeddings)
                    if similarity is not None:
                        st.write(f"Cosine Similarity between '{word1}' and '{word2}': {similarity:.4f}")
                    else:
                        st.write("Words not found in vocabulary")
            
            except Exception as e:
                st.error("Embedding generation failed:")
                st.error(str(e))
    
    with tab5:
        st.header("📖 Language Model Text Generation")
        display_learning_note("""
        A simple language model predicts the next token based on previous context. 
        Temperature controls the randomness of generation.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_length = st.slider("Generation Length", 10, 200, 50)
        
        with col2:
            temperature = st.slider("Temperature", 0.1, 1.0, 0.7)
        
        if st.button("Generate Text"):
            try:
                # Build vocabulary
                vocab = build_vocabulary(
                    [input_text], 
                    tokenizer=advanced_tokenize, 
                    max_size=1000,
                    add_special_tokens=True
                )
                
                # Initialize model
                model = SimpleLanguageModel(
                    vocab_size=len(vocab),
                    embedding_dim=128,
                    hidden_dim=256,
                    num_layers=2
                )
                
                # Generate text
                generated_text = generate_text(
                    model, 
                    prefix=input_text, 
                    vocab=vocab, 
                    max_length=max_length, 
                    temperature=temperature
                )
                
                st.write("Generated Text:")
                st.write(generated_text)
                
                st.info("""
                Note: This is a simple model trained on minimal data. 
                Real-world language models use much larger datasets and more complex architectures.
                """)
            
            except Exception as e:
                st.error("Text generation failed:")
                st.error(str(e))
    
    with tab6:
        st.header("📊 Token Distribution")
        display_learning_note("""
        Token frequency analysis reveals the most common words in your text. 
        This helps understand the vocabulary composition.
        """)
        
        if st.button("Plot Token Frequencies"):
            try:
                # Plot token distribution
                fig = plot_token_distribution([input_text])
                st.pyplot(fig)
                plt.close(fig)
                
                st.info("Bars represent the most frequent tokens in the input text.")
            
            except Exception as e:
                st.error("Token distribution visualization failed:")
                st.error(str(e))

    # Footer with learning journey context
    st.markdown("---")
    st.markdown("""
    ### 🚀 Continuing Your NLP Learning Journey
    
    This interactive pipeline demonstrates core NLP concepts. Next steps:
    - Explore more advanced tokenization strategies
    - Experiment with different embedding techniques
    - Build more sophisticated language models
    
    *Part of the hands-on LLM learning series*
    """)

# Run the app
if __name__ == "__main__":
    main()