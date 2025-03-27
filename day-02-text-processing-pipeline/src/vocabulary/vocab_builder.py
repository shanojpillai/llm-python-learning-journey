"""
Vocabulary building utilities for text processing.
"""
from collections import Counter

def build_vocabulary(texts, tokenizer, max_size=None, min_freq=1, add_special_tokens=False):
    """
    Build a vocabulary from a list of texts.
    
    Args:
        texts (list): List of text strings
        tokenizer (function): Tokenization function to use
        max_size (int, optional): Maximum vocabulary size
        min_freq (int, optional): Minimum frequency for a token to be included
        add_special_tokens (bool): Whether to add special tokens to the vocabulary
        
    Returns:
        dict: Mapping of tokens to their IDs
    """
    # Tokenize all texts
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenizer(text))
    
    # Count token frequencies
    token_counts = Counter(all_tokens)
    
    # Filter by minimum frequency
    token_counts = {token: count for token, count in token_counts.items() 
                   if count >= min_freq}
    
    # Sort by frequency (descending)
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Limit vocabulary size if specified
    if max_size is not None:
        special_token_count = 4 if add_special_tokens else 0  # <|unk|>, <|endoftext|>, [BOS], [EOS]
        sorted_tokens = sorted_tokens[:max_size - special_token_count]
    
    # Create vocabulary with token IDs
    vocab = {}
    
    # Add special tokens first if requested
    if add_special_tokens:
        vocab['<|unk|>'] = 0  # Unknown token
        vocab['<|endoftext|>'] = 1  # End of text
        vocab['[BOS]'] = 2  # Beginning of sequence
        vocab['[EOS]'] = 3  # End of sequence
        vocab['[PAD]'] = 4  # Padding token
        next_id = 5
    else:
        next_id = 0
    
    # Add regular tokens
    for token, _ in sorted_tokens:
        if token not in vocab:  # Avoid duplicates with special tokens
            vocab[token] = next_id
            next_id += 1
    
    return vocab

def tokenize_and_encode(text, vocab, tokenizer):
    """
    Tokenize text and convert to token IDs.
    
    Args:
        text (str): Input text
        vocab (dict): Vocabulary mapping tokens to IDs
        tokenizer (function): Tokenization function
        
    Returns:
        tuple: (tokens, token_ids)
    """
    # Tokenize the text
    tokens = tokenizer(text)
    
    # Convert tokens to IDs
    token_ids = []
    for token in tokens:
        # Use the token ID if in vocabulary, otherwise use unknown token ID
        if token in vocab:
            token_ids.append(vocab[token])
        elif '<|unk|>' in vocab:
            token_ids.append(vocab['<|unk|>'])
        else:
            # If no unknown token, use a placeholder
            token_ids.append(-1)
    
    return tokens, token_ids

def decode_token_ids(token_ids, vocab):
    """
    Convert token IDs back to text.
    
    Args:
        token_ids (list): List of token IDs
        vocab (dict): Vocabulary mapping tokens to IDs
        
    Returns:
        str: Decoded text
    """
    # Create reverse mapping (ID to token)
    id_to_token = {id: token for token, id in vocab.items()}
    
    # Convert IDs back to tokens
    tokens = []
    for id in token_ids:
        if id in id_to_token:
            token = id_to_token[id]
            # Skip special tokens in the output
            if token not in ['<|unk|>', '<|endoftext|>', '[BOS]', '[EOS]', '[PAD]']:
                tokens.append(token)
        else:
            tokens.append("<UNKNOWN>")
    
    # Join tokens (simple approach - doesn't handle punctuation spacing perfectly)
    return ' '.join(tokens)