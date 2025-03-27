"""
Simple language model implementation for text generation.
"""
import torch
import torch.nn as nn
import numpy as np
from src.preprocessing.tokenization import advanced_tokenize
from src.vocabulary.vocab_builder import tokenize_and_encode, decode_token_ids

class SimpleLanguageModel(nn.Module):
    """
    A simple language model that predicts the next token based on previous tokens.
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        """
        Initialize the language model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            embedding_dim (int): Dimension of the token embeddings
            hidden_dim (int): Dimension of the hidden layers
            num_layers (int): Number of recurrent layers
            dropout (float): Dropout probability
        """
        super(SimpleLanguageModel, self).__init__()
        
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Store model parameters
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(self, x, hidden=None):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of token IDs with shape (batch_size, sequence_length)
            hidden (tuple, optional): Initial hidden state
            
        Returns:
            tuple: (output, hidden_state)
        """
        # Convert input to embeddings
        embedded = self.embedding(x)
        
        # Pass through LSTM
        output, hidden = self.lstm(embedded, hidden)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Project to vocabulary size
        output = self.fc(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        """
        Initialize hidden state.
        
        Args:
            batch_size (int): Batch size
            
        Returns:
            tuple: (hidden, cell) state tuple
        """
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden

def generate_text(model, prefix, vocab, max_length=50, temperature=1.0):
    """
    Generate text using the trained language model.
    
    Args:
        model (SimpleLanguageModel): The language model
        prefix (str): Starting text for generation
        vocab (dict): Vocabulary mapping
        max_length (int): Maximum length of generated text
        temperature (float): Controls randomness (higher = more random)
        
    Returns:
        str: Generated text
    """
    model.eval()  # Set model to evaluation mode
    
    # Tokenize and encode the prefix
    tokens, token_ids = tokenize_and_encode(prefix, vocab, tokenizer=advanced_tokenize)
    
    # Convert to tensor
    input_ids = torch.tensor(token_ids).unsqueeze(0)  # Add batch dimension
    
    # Generate new token IDs
    generated_ids = token_ids.copy()
    
    # Hidden state
    hidden = None
    
    # Generate tokens one by one
    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            output, hidden = model(input_ids, hidden)
            
            # Get the predictions for the last token
            pred = output[0, -1, :].div(temperature).exp()
            
            # Convert to probability distribution
            pred = pred / pred.sum()
            
            # Sample from the distribution
            next_token_id = torch.multinomial(pred, 1).item()
            
            # Add to generated IDs
            generated_ids.append(next_token_id)
            
            # Stop if we generate end of text token
            if '<|endoftext|>' in vocab and next_token_id == vocab['<|endoftext|>']:
                break
                
            # Update input for next iteration
            input_ids = torch.tensor([[next_token_id]]).long()
    
    # Decode the generated IDs
    generated_text = decode_token_ids(generated_ids, vocab)
    
    return generated_text

def train_model(model, dataset, epochs=5, batch_size=32, learning_rate=0.001):
    """
    Train the language model.
    
    Args:
        model (SimpleLanguageModel): The language model
        dataset (list): List of (input, target) pairs
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        learning_rate (float): Learning rate
        
    Returns:
        list: Training losses
    """
    model.train()  # Set model to training mode
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training losses
    losses = []
    
    # Training loop
    for epoch in range(epochs):
        # Shuffle dataset
        np.random.shuffle(dataset)
        
        # Process in batches
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            
            # Prepare batch
            inputs = [item[0] for item in batch]
            targets = [item[1] for item in batch]
            
            # Convert to tensors
            input_tensor = torch.tensor(inputs).long()
            target_tensor = torch.tensor(targets).long()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output, _ = model(input_tensor)
            
            # Reshape output and target for loss calculation
            output = output.view(-1, model.vocab_size)
            target_tensor = target_tensor.view(-1)
            
            # Calculate loss
            loss = criterion(output, target_tensor)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Record loss
            losses.append(loss.item())
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(losses[-len(dataset)//batch_size:]):.4f}")
    
    return losses