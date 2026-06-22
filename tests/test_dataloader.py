"""Tests for dataloader module."""
import unittest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataloader import LiteGPTDataLoader


class TestLiteGPTDataLoader(unittest.TestCase):
    """Test dataloader functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.batch_size = 32
        cls.seq_len = 128
    
    def test_dataloader_initialization(self):
        """Test dataloader initializes correctly."""
        loader = LiteGPTDataLoader(split="train")
        self.assertIsNotNone(loader)
    
    def test_get_batch_shape(self):
        """Test batch has correct shape."""
        loader = LiteGPTDataLoader(split="train")
        x, y = loader.get_batch()
        
        # Check shapes
        self.assertEqual(len(x.shape), 2)  # Should be 2D tensor
        self.assertEqual(len(y.shape), 2)  # Should be 2D tensor
        
        # Check first dimension is batch size
        self.assertEqual(x.shape[0], loader.batch_size)
        self.assertEqual(y.shape[0], loader.batch_size)
        
        # Check second dimension is sequence length
        self.assertEqual(x.shape[1], loader.seq_len)
        self.assertEqual(y.shape[1], loader.seq_len)
    
    def test_get_batch_dtype(self):
        """Test batch has correct dtype."""
        loader = LiteGPTDataLoader(split="train")
        x, y = loader.get_batch()
        
        self.assertEqual(x.dtype, torch.long)
        self.assertEqual(y.dtype, torch.long)
    
    def test_get_batch_values(self):
        """Test batch values are in valid range."""
        loader = LiteGPTDataLoader(split="train")
        x, y = loader.get_batch()
        
        # Tokens should be non-negative
        self.assertTrue(torch.all(x >= 0))
        self.assertTrue(torch.all(y >= 0))
    
    def test_multiple_batches(self):
        """Test getting multiple batches."""
        loader = LiteGPTDataLoader(split="train")
        
        batch1_x, batch1_y = loader.get_batch()
        batch2_x, batch2_y = loader.get_batch()
        
        # Different batches should not be identical
        # (with very high probability)
        self.assertFalse(torch.equal(batch1_x, batch2_x))
    
    def test_different_splits(self):
        """Test different data splits."""
        splits = ["train", "val"]
        
        for split in splits:
            with self.subTest(split=split):
                loader = LiteGPTDataLoader(split=split)
                x, y = loader.get_batch()
                
                self.assertEqual(x.shape[0], loader.batch_size)
                self.assertEqual(x.shape[1], loader.seq_len)
    
    def test_batch_size_attribute(self):
        """Test batch size attribute."""
        loader = LiteGPTDataLoader(split="train")
        self.assertGreater(loader.batch_size, 0)
        self.assertIsInstance(loader.batch_size, int)
    
    def test_seq_len_attribute(self):
        """Test sequence length attribute."""
        loader = LiteGPTDataLoader(split="train")
        self.assertGreater(loader.seq_len, 0)
        self.assertIsInstance(loader.seq_len, int)
    
    def test_consecutive_batches_different(self):
        """Test that consecutive batches are different."""
        loader = LiteGPTDataLoader(split="train")
        
        batches = [loader.get_batch() for _ in range(5)]
        
        # Check that batches are different
        for i in range(len(batches) - 1):
            self.assertFalse(torch.equal(batches[i][0], batches[i + 1][0]))


if __name__ == "__main__":
    unittest.main()
