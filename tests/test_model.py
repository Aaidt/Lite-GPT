"""Tests for model module."""
import unittest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.model import LiteGPT
from data.dataloader import LiteGPTDataLoader


class TestLiteGPT(unittest.TestCase):
    """Test LiteGPT model."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.model = LiteGPT()
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model.to(cls.device)
        cls.dataloader = LiteGPTDataLoader(split="train")
    
    def test_model_initialization(self):
        """Test model initializes correctly."""
        model = LiteGPT()
        self.assertIsNotNone(model)
    
    def test_model_forward_pass(self):
        """Test forward pass works."""
        x, y = self.dataloader.get_batch()
        x, y = x.to(self.device), y.to(self.device)
        
        logits, loss = self.model(x, y)
        
        self.assertIsNotNone(logits)
        self.assertIsNotNone(loss)
    
    def test_logits_shape(self):
        """Test logits have correct shape."""
        x, y = self.dataloader.get_batch()
        x, y = x.to(self.device), y.to(self.device)
        
        logits, _ = self.model(x, y)
        
        # Logits should have shape (batch_size, seq_len, vocab_size)
        self.assertEqual(len(logits.shape), 3)
        self.assertEqual(logits.shape[0], x.shape[0])
        self.assertEqual(logits.shape[1], x.shape[1])
    
    def test_loss_shape(self):
        """Test loss is scalar."""
        x, y = self.dataloader.get_batch()
        x, y = x.to(self.device), y.to(self.device)
        
        _, loss = self.model(x, y)
        
        # Loss should be scalar
        self.assertEqual(loss.shape, torch.Size([]))
    
    def test_loss_value_range(self):
        """Test loss is in reasonable range."""
        x, y = self.dataloader.get_batch()
        x, y = x.to(self.device), y.to(self.device)
        
        _, loss = self.model(x, y)
        
        # Loss should be positive
        self.assertGreater(loss.item(), 0)
        # Loss shouldn't be inf or nan
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
    
    def test_inference_mode(self):
        """Test forward pass with no labels (inference)."""
        x, _ = self.dataloader.get_batch()
        x = x.to(self.device)
        
        logits, loss = self.model(x, None)
        
        self.assertIsNotNone(logits)
        self.assertIsNone(loss)
    
    def test_model_device_movement(self):
        """Test moving model to different devices."""
        model = LiteGPT()
        
        # Test CPU
        model.to("cpu")
        x, y = self.dataloader.get_batch()
        x, y = x.to("cpu"), y.to("cpu")
        logits, loss = model(x, y)
        self.assertIsNotNone(loss)
    
    def test_parameter_count(self):
        """Test model has reasonable number of parameters."""
        model = LiteGPT()
        
        param_count = sum(p.numel() for p in model.parameters())
        self.assertGreater(param_count, 0)
    
    def test_gradient_flow(self):
        """Test gradients flow through model."""
        model = LiteGPT()
        model.to(self.device)
        x, y = self.dataloader.get_batch()
        x, y = x.to(self.device), y.to(self.device)
        
        logits, loss = model(x, y)
        loss.backward()
        
        # Check that at least some gradients are computed
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        self.assertTrue(has_gradients)
    
    def test_deterministic_behavior(self):
        """Test model produces same output for same input."""
        torch.manual_seed(42)
        model = LiteGPT()
        model.to(self.device)
        model.eval()
        
        x = torch.randint(0, 100, (8, 128)).to(self.device)
        
        with torch.no_grad():
            logits1, _ = model(x, None)
            logits2, _ = model(x, None)
        
        self.assertTrue(torch.allclose(logits1, logits2))
    
    def test_different_batch_sizes(self):
        """Test model works with different batch sizes."""
        model = LiteGPT()
        model.to(self.device)
        
        for batch_size in [1, 4, 8, 16]:
            x = torch.randint(0, 100, (batch_size, 128)).to(self.device)
            y = torch.randint(0, 100, (batch_size, 128)).to(self.device)
            
            with self.subTest(batch_size=batch_size):
                logits, loss = model(x, y)
                self.assertEqual(logits.shape[0], batch_size)
                self.assertIsNotNone(loss)
    
    def test_output_dtypes(self):
        """Test output dtypes are correct."""
        x, y = self.dataloader.get_batch()
        x, y = x.to(self.device), y.to(self.device)
        
        logits, loss = self.model(x, y)
        
        self.assertEqual(logits.dtype, torch.float32)
        self.assertEqual(loss.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
