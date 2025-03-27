"""
Tokenization utilities for text processing.
"""
import re
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def basic_word_tokenize(text):
    """
    Basic word-level tokenization using NLTK's word_tokenize.
    
    Args:
        text (str): The input text to tokenize
        
    Returns:
        list: List of word tokens
    """
    return word_tokenize(text)

def advanced_tokenize(text):
    """
    Advanced tokenization that handles punctuation and special characters separately.
    
    Args:
        text (str): The input text to tokenize
        
    Returns:
        list: List of tokens including separated punctuation
    """
    # First, separate punctuation from words
    text = re.sub(r'([^\w\s])', r' \1 ', text)
    
    # Split by whitespace
    tokens = text.split()
    
    # Remove empty tokens
    tokens = [token for token in tokens if token.strip()]
    
    return tokens

def character_tokenize(text):
    """
    Character-level tokenization.
    
    Args:
        text (str): The input text to tokenize
        
    Returns:
        list: List of character tokens
    """
    return list(text)

def visualize_tokenization_comparison(text):
    """
    Create a visualization comparing different tokenization strategies.
    
    Args:
        text (str): Text to tokenize using different methods
        
    Returns:
        matplotlib.figure.Figure: Figure object with the visualization
    """
    # Tokenize using different methods
    word_tokens = basic_word_tokenize(text)
    advanced_tokens = advanced_tokenize(text)
    char_tokens = character_tokenize(text)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    methods = ['Word', 'Advanced', 'Character']
    token_counts = [len(word_tokens), len(advanced_tokens), len(char_tokens)]
    
    bars = ax.bar(methods, token_counts, color=['#3498db', '#2ecc71', '#e74c3c'])
    
    # Add token count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Customize the plot
    ax.set_title('Comparison of Tokenization Methods', fontsize=15)
    ax.set_ylabel('Number of Tokens', fontsize=12)
    ax.set_ylim(0, max(token_counts) * 1.2)  # Add some space for labels
    
    # Add a grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Show sample tokens in text boxes
    textstr = (
        f'Word tokens (first 3): {word_tokens[:3]}\n\n'
        f'Advanced tokens (first 3): {advanced_tokens[:3]}\n\n'
        f'Character tokens (first 5): {char_tokens[:5]}'
    )
    
    # Add text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig