"""
Tests for the preprocessing module.
"""
import unittest
import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing.cleaner import clean_text, remove_special_chars
from src.preprocessing.tokenization import basic_word_tokenize, advanced_tokenize, character_tokenize

class TestCleaner(unittest.TestCase):
    """Test cases for text cleaning functions."""
    
    def test_clean_text(self):
        """Test the clean_text function."""
        text = "  Hello, World!  "
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "hello, world!")
    
    def test_remove_special_chars(self):
        """Test the remove_special_chars function."""
        text = "Hello, World! This is a test."
        cleaned = remove_special_chars(text)
        self.assertEqual(cleaned, "Hello World This is a test")

class TestTokenization(unittest.TestCase):
    """Test cases for tokenization functions."""
    
    def test_basic_word_tokenize(self):
        """Test the basic_word_tokenize function."""
        text = "Hello, world!"
        tokens = basic_word_tokenize(text)
        self.assertEqual(tokens, ['Hello', ',', 'world', '!'])
    
    def test_advanced_tokenize(self):
        """Test the advanced_tokenize function."""
        text = "Hello, world!"
        tokens = advanced_tokenize(text)
        self.assertEqual(tokens, ['Hello', ',', 'world', '!'])
    
    def test_character_tokenize(self):
        """Test the character_tokenize function."""
        text = "Hello"
        tokens = character_tokenize(text)
        self.assertEqual(tokens, ['H', 'e', 'l', 'l', 'o'])
    
    def test_tokenize_empty_string(self):
        """Test tokenizing an empty string."""
        text = ""
        word_tokens = basic_word_tokenize(text)
        advanced_tokens = advanced_tokenize(text)
        char_tokens = character_tokenize(text)
        
        self.assertEqual(word_tokens, [])
        self.assertEqual(advanced_tokens, [])
        self.assertEqual(char_tokens, [])
    
    def test_tokenize_with_numbers(self):
        """Test tokenizing text with numbers."""
        text = "Testing 123 numbers."
        tokens = advanced_tokenize(text)
        self.assertIn('123', tokens)

if __name__ == '__main__':
    unittest.main()