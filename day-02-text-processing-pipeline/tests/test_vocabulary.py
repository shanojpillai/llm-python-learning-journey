"""
Tests for the vocabulary module.
"""
import unittest
import sys
import os

# Add the parent directory to the path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vocabulary.vocab_builder import build_vocabulary, tokenize_and_encode, decode_token_ids
from src.preprocessing.tokenization import advanced_tokenize

class TestVocabularyBuilder(unittest.TestCase):
    """Test cases for vocabulary building functions."""
    
    def setUp(self):
        """Set up test data."""
        self.texts = [
            "This is a test.",
            "Another test sentence.",
            "Testing vocabulary building."
        ]
    
    def test_build_vocabulary(self):
        """Test building a vocabulary from texts."""
        vocab = build_vocabulary(self.texts, tokenizer=advanced_tokenize)
        
        # Check that the vocabulary contains all unique tokens
        all_tokens = []
        for text in self.texts:
            all_tokens.extend(advanced_tokenize(text))
        
        unique_tokens = set(all_tokens)
        
        self.assertEqual(len(vocab), len(unique_tokens))
        
        # Check that all tokens are in the vocabulary
        for token in unique_tokens:
            self.assertIn(token, vocab)
    
    def test_build_vocabulary_with_special_tokens(self):
        """Test building a vocabulary with special tokens."""
        vocab = build_vocabulary(self.texts, tokenizer=advanced_tokenize, add_special_tokens=True)
        
        # Check that special tokens are in the vocabulary
        special_tokens = ['<|unk|>', '<|endoftext|>', '[BOS]', '[EOS]', '[PAD]']
        for token in special_tokens:
            self.assertIn(token, vocab)
    
    def test_tokenize_and_encode(self):
        """Test tokenizing and encoding text."""
        vocab = build_vocabulary(self.texts, tokenizer=advanced_tokenize)
        
        text = "This is a test."
        tokens, token_ids = tokenize_and_encode(text, vocab, advanced_tokenize)
        
        # Check that the number of tokens matches the number of IDs
        self.assertEqual(len(tokens), len(token_ids))
        
        # Check that each token ID corresponds to the right token
        for token, token_id in zip(tokens, token_ids):
            self.assertEqual(vocab[token], token_id)
    
    def test_decode_token_ids(self):
        """Test decoding token IDs back to text."""
        vocab = build_vocabulary(self.texts, tokenizer=advanced_tokenize)
        
        text = "This is a test."
        _, token_ids = tokenize_and_encode(text, vocab, advanced_tokenize)
        
        decoded_text = decode_token_ids(token_ids, vocab)
        
        # The decoded text won't exactly match the original due to spacing
        # but it should contain all the same tokens
        self.assertIn("This", decoded_text)
        self.assertIn("is", decoded_text)
        self.assertIn("a", decoded_text)
        self.assertIn("test", decoded_text)

if __name__ == '__main__':
    unittest.main()