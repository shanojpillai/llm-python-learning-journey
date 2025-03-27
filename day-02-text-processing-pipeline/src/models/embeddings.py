"""
Word embedding utilities for creating and visualizing word embeddings.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def create_embeddings(vocab, embedding_dim=100, seed=42):
    """
    Create random word embeddings for a vocabulary.
    
    Args:
        vocab (dict): Vocabulary mapping tokens to IDs
        embedding_dim (int): Dimension of the embeddings
        seed (int): Random seed for reproducibility
        
    Returns:
        torch.Tensor: Embedding matrix of shape (vocab_size, embedding_dim)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create random embeddings
    vocab_size = len(vocab)
    embeddings = torch.randn(vocab_size, embedding_dim)
    
    # Normalize embeddings to have unit length
    norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
    embeddings = embeddings / norms
    
    return embeddings

def visualize_embeddings(vocab, embeddings, method='PCA', top_n=100):
    """
    Visualize word embeddings in 2D.
    
    Args:
        vocab (dict): Vocabulary mapping tokens to IDs
        embeddings (torch.Tensor): Embedding matrix
        method (str): Dimensionality reduction method ('PCA' or 't-SNE')
        top_n (int): Number of most frequent words to visualize
        
    Returns:
        matplotlib.figure.Figure: Figure object with the visualization
    """
    # Convert embeddings to numpy array
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().numpy()
    
    # Take the top N words (assuming lower IDs are more frequent)
    tokens = []
    token_ids = []
    
    # Filter out special tokens
    special_tokens = ['<|unk|>', '<|endoftext|>', '[BOS]', '[EOS]', '[PAD]']
    
    # Sort vocabulary by ID
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    for token, id in sorted_vocab:
        if token not in special_tokens and id < len(embeddings):
            tokens.append(token)
            token_ids.append(id)
            if len(tokens) >= top_n:
                break
    
    # Get embeddings for selected tokens
    selected_embeddings = embeddings[token_ids]
    
    # Reduce dimensionality to 2D
    if method == 'PCA':
        reducer = PCA(n_components=2)
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=42)
    
    reduced_embeddings = reducer.fit_transform(selected_embeddings)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot points
    ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='blue', alpha=0.5)
    
    # Add labels for some words
    num_labels = min(25, len(tokens))  # Limit the number of labels to avoid crowding
    step = max(1, len(tokens) // num_labels)
    
    for i in range(0, len(tokens), step):
        ax.annotate(tokens[i], 
                   (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                   fontsize=9,
                   alpha=0.8)
    
    # Set title and labels
    ax.set_title(f'2D Visualization of Word Embeddings using {method}', fontsize=15)
    ax.set_xlabel(f'{method} Component 1', fontsize=12)
    ax.set_ylabel(f'{method} Component 2', fontsize=12)
    
    # Remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    
    return fig

def calculate_similarity(word1, word2, vocab, embeddings):
    """
    Calculate cosine similarity between two word embeddings.
    
    Args:
        word1 (str): First word
        word2 (str): Second word
        vocab (dict): Vocabulary mapping tokens to IDs
        embeddings (torch.Tensor): Embedding matrix
        
    Returns:
        float: Cosine similarity between the two word embeddings
    """
    # Check if words are in vocabulary
    if word1 not in vocab or word2 not in vocab:
        return None
    
    # Get embeddings
    id1 = vocab[word1]
    id2 = vocab[word2]
    
    emb1 = embeddings[id1]
    emb2 = embeddings[id2]
    
    # Calculate cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
    
    return cos_sim.item()