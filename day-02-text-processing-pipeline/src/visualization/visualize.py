"""
Visualization utilities for text processing.
"""
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from src.preprocessing.tokenization import advanced_tokenize
from src.vocabulary.vocab_builder import build_vocabulary

def plot_token_distribution(texts, tokenizer=advanced_tokenize, top_n=20):
    """
    Plot distribution of token frequencies.
    
    Args:
        texts (list): List of text strings
        tokenizer (function): Tokenization function
        top_n (int): Number of most frequent tokens to display
        
    Returns:
        matplotlib.figure.Figure: Figure with the plot
    """
    # Tokenize all texts
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenizer(text))
    
    # Count token frequencies
    token_counts = Counter(all_tokens)
    
    # Get the most common tokens
    most_common = token_counts.most_common(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot bar chart
    tokens, counts = zip(*most_common)
    ax.bar(range(len(tokens)), counts, color='skyblue')
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    
    # Add count labels on top of bars
    for i, count in enumerate(counts):
        ax.text(i, count + 0.5, str(count), ha='center')
    
    # Customize the plot
    ax.set_title(f'Top {top_n} Most Frequent Tokens', fontsize=15)
    ax.set_xlabel('Token', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig