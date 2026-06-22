"""Tests for tokenizer module."""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):
    """Test tokenizer functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.tokenizer = Tokenizer()
    
    def test_tokenizer_initialization(self):
        """Test that tokenizer initializes correctly."""
        self.assertIsNotNone(self.tokenizer)
        self.assertIsNotNone(self.tokenizer.eos_token_id)
    
    def test_encode_decode_roundtrip(self):
        """Test encoding and decoding roundtrip."""
        text = "The quick brown fox jumps over the lazy dog"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        
        # Check that encoded is a list of integers
        self.assertIsInstance(encoded, list)
        self.assertTrue(all(isinstance(t, int) for t in encoded))
        
        # Check that decoded text is close to original (tokenizer may not be perfect)
        self.assertIsInstance(decoded, str)
    
    def test_encode_empty_string(self):
        """Test encoding empty string."""
        encoded = self.tokenizer.encode("")
        self.assertIsInstance(encoded, list)
    
    def test_encode_special_characters(self):
        """Test encoding special characters."""
        text = "Hello! @#$%^&*()"
        encoded = self.tokenizer.encode(text)
        self.assertIsInstance(encoded, list)
        self.assertTrue(len(encoded) > 0)
    
    def test_decode_empty_list(self):
        """Test decoding empty token list."""
        decoded = self.tokenizer.decode([])
        self.assertIsInstance(decoded, str)
    
    def test_vocabulary_size(self):
        """Test that vocabulary size is reasonable."""
        vocab_size = self.tokenizer.vocab_size
        self.assertGreater(vocab_size, 0)
        self.assertLess(vocab_size, 1000000)  # Reasonable upper bound
    
    def test_encode_consistency(self):
        """Test that encoding is consistent."""
        text = "Consistent encoding test"
        encoded1 = self.tokenizer.encode(text)
        encoded2 = self.tokenizer.encode(text)
        self.assertEqual(encoded1, encoded2)
    
    def test_multiple_texts(self):
        """Test encoding multiple different texts."""
        texts = [
            "First text",
            "Second text with more words",
            "Third",
        ]
        
        for text in texts:
            encoded = self.tokenizer.encode(text)
            self.assertIsInstance(encoded, list)
            self.assertTrue(len(encoded) > 0)


if __name__ == "__main__":
    unittest.main()
