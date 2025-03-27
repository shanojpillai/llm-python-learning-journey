"""
Tests for the language model module.
"""
import unittest
import sys
import os

# Add the parent directory to the path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models.language_model import SimpleLanguageModel
from src.vocabulary.vocab_builder import build_vocabulary
from src.preprocessing.tokenization import advanced_tokenize

class TestLanguageModel(unittest.TestCase):
    """Test cases for language model."""
    
    def setUp(self):
        """Set up test data."""
        self.texts = [
            "This is a test.",
            "Another test sentence.",
            "Testing language model."
        ]
        self.vocab = build_vocabulary(self.texts, tokenizer=advanced_tokenize, add_special_tokens=True)
        self.vocab_size = len(self.vocab)
        self.embedding_dim = 10
        self.hidden_dim = 20
        self.num_layers = 2
    
    def test_model_initialization(self):
        """Test language model initialization."""
        model = SimpleLanguageModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )
        
        # Check model components
        self.assertEqual(model.embedding.embedding_dim, self.embedding_dim)
        self.assertEqual(model.embedding.num_embeddings, self.vocab_size)
        self.assertEqual(model.lstm.hidden_size, self.hidden_dim)
        self.assertEqual(model.lstm.num_layers, self.num_layers)
        self.assertEqual(model.fc.out_features, self.vocab_size)
    
    def test_model_forward(self):
        """Test forward pass through the model."""
        model = SimpleLanguageModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )
        
        # Create dummy input
        batch_size = 2
        seq_length = 5
        x = torch.randint(0, self.vocab_size, (batch_size, seq_length))
        
        # Forward pass
        output, hidden = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_length, self.vocab_size))
        
        # Check hidden state shape
        self.assertEqual(hidden[0].shape, (self.num_layers, batch_size, self.hidden_dim))
        self.assertEqual(hidden[1].shape, (self.num_layers, batch_size, self.hidden_dim))

if __name__ == '__main__':
    unittest.main()