"""
Tests for the embeddings module.
"""
import unittest
import sys
import os

# Add the parent directory to the path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models.embeddings import create_embeddings, calculate_similarity
from src.vocabulary.vocab_builder import build_vocabulary
from src.preprocessing.tokenization import advanced_tokenize

class TestEmbeddings(unittest.TestCase):
    """Test cases for embedding functions."""
    
    def setUp(self):
        """Set up test data."""
        self.texts = [
            "This is a test.",
            "Another test sentence.",
            "Testing embeddings."
        ]
        self.vocab = build_vocabulary(self.texts, tokenizer=advanced_tokenize)
        self.embedding_dim = 10
    
    def test_create_embeddings(self):
        """Test creating embeddings."""
        embeddings = create_embeddings(self.vocab, self.embedding_dim)
        
        # Check shape
        self.assertEqual(embeddings.shape, (len(self.vocab), self.embedding_dim))
        
        # Check normalization (length should be approximately 1)
        for i in range(len(self.vocab)):
            vec_norm = torch.norm(embeddings[i]).item()
            self.assertAlmostEqual(vec_norm, 1.0, places=5)
    
    def test_calculate_similarity(self):
        """Test calculating similarity between word embeddings."""
        embeddings = create_embeddings(self.vocab, self.embedding_dim)
        
        # Add some words to test
        if 'test' in self.vocab and 'testing' in self.vocab:
            similarity = calculate_similarity('test', 'testing', self.vocab, embeddings)
            
            # Similarity should be between -1 and 1
            self.assertGreaterEqual(similarity, -1.0)
            self.assertLessEqual(similarity, 1.0)
        
        # Test words not in vocabulary
        similarity = calculate_similarity('nonexistent', 'word', self.vocab, embeddings)
        self.assertIsNone(similarity)

if __name__ == '__main__':
    unittest.main()