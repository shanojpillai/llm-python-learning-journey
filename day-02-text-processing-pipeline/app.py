"""
Streamlit application for text processing pipeline.
"""
import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

# Add the src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.cleaner import clean_text
from src.preprocessing.tokenization import (
    basic_word_tokenize,
    advanced_tokenize,
    visualize_tokenization_comparison
)
from src.vocabulary.vocab_builder import (
    build_vocabulary,
    tokenize_and_encode,
    decode_token_ids
)
from src.models.embeddings import create_embeddings, visualize_embeddings
from src.models.language_model import SimpleLanguageModel, generate_text
from src.visualization.visualize import plot_token_distribution

st.set_page_config(
    page_title="Text Processing Pipeline",
    page_icon="📚",
    layout="wide"
)

st.title("Text Processing Pipeline for Language Models")

st.markdown("""
This application demonstrates a complete text processing pipeline for language models.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Introduction", "Tokenization", "Vocabulary Building", 
     "Token IDs & Special Tokens", "Word Embeddings", "Mini Language Model"]
)

# Create a placeholder for sample data
@st.cache_data
def load_sample_data():
    try:
        # Try to load sample headlines
        with open("data/raw/sample_headlines.txt", "r") as f:
            headlines = [line.strip() for line in f.readlines() if line.strip()]
    except:
        # If file doesn't exist, use default sample data
        headlines = [
            "Scientists discover new planet in nearby solar system",
            "Local man finds $10,000 in old couch",
            "World leaders agree on climate change action plan",
            "Study shows coffee may help prevent certain diseases",
            "Tech company unveils revolutionary new smartphone",
            "Stock market reaches all-time high amid economic recovery"
        ]
    
    return headlines

sample_data = load_sample_data()

# Introduction page
if page == "Introduction":
    st.header("Introduction to Text Processing for LLMs")
    
    st.markdown("""
    ### Key Concepts We'll Explore:
    
    1. **Tokenization**: Breaking down text into smaller units (words, subwords, characters)
    2. **Vocabulary Building**: Creating a list of unique tokens from our text
    3. **Converting Tokens to IDs**: Transforming tokens into numerical representations
    4. **Special Context Tokens**: Adding special tokens to provide context
    5. **Word Embeddings**: Creating vector representations of words
    6. **Simple Language Model**: Building a basic model that generates text
    
    This interactive application will guide you through each of these concepts with practical examples.
    """)

# Tokenization page
elif page == "Tokenization":
    st.header("Tokenization")
    
    st.markdown("""
    **Tokenization** is the process of breaking text into smaller units called tokens.
    These tokens can be words, subwords, or characters. The choice of tokenization
    strategy significantly impacts the vocabulary size and model performance.
    
    Let's explore different tokenization approaches:
    """)
    
    # Sample text for tokenization demo
    sample_text = st.text_area(
        "Enter some text to tokenize",
        "Hello, world! This is a test. Can tokenization handle punctuation and special-characters?",
        height=100
    )
    
    st.subheader("1. Basic Word Tokenization")
    st.markdown("""
    Word-level tokenization splits text by whitespace and treats punctuation as part of words.
    This is simple but can lead to a large vocabulary and doesn't handle morphological variations well.
    """)
    
    word_tokens = basic_word_tokenize(sample_text)
    st.write(f"Word Tokens ({len(word_tokens)} tokens):", word_tokens)
    
    st.subheader("2. Advanced Tokenization")
    st.markdown("""
    This approach handles special characters, punctuation, and whitespace as separate tokens.
    It provides more granular control over the tokenization process.
    """)
    
    advanced_tokens = advanced_tokenize(sample_text)
    st.write(f"Advanced Tokens ({len(advanced_tokens)} tokens):", advanced_tokens)
    
    # Tokenization comparison chart
    st.subheader("Tokenization Comparison")
    fig = visualize_tokenization_comparison(sample_text)
    st.pyplot(fig)
    
    # Add the remaining pages for vocabulary building, token IDs, word embeddings, and language model
    # These can be implemented as you progress through the project

if __name__ == "__main__":
    # This will only run when the script is executed directly
    pass