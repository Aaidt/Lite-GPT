from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TrainingMetrics:
    
    losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    tokens_per_sec: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    perplexity: List[float] = field(default_factory=list)
    
    def add_train_step(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: float,
        tokens_per_sec: float,
    ):
        """Add training step metrics."""
        self.steps.append(step)
        self.losses.append(loss)
        self.learning_rates.append(lr)
        self.grad_norms.append(grad_norm)
        self.tokens_per_sec.append(tokens_per_sec)
    
    def add_val_loss(self, val_loss: float):
        """Add validation loss."""
        self.val_losses.append(val_loss)
    
    def add_perplexity(self, perplexity: float):
        """Add perplexity."""
        self.perplexity.append(perplexity)
    
    def get_latest_metrics(self) -> Dict:
        """Get the latest metrics as a dictionary."""
        metrics = {
            "loss": self.losses[-1] if self.losses else 0.0,
            "learning_rate": self.learning_rates[-1] if self.learning_rates else 0.0,
            "grad_norm": self.grad_norms[-1] if self.grad_norms else 0.0,
            "tokens_per_sec": self.tokens_per_sec[-1] if self.tokens_per_sec else 0.0,
            "perplexity": self.perplexity[-1] if self.perplexity else 0.0,
        }
        if self.val_losses:
            metrics["val_loss"] = self.val_losses[-1]
        return metrics
    
    def get_avg_metrics(self, window: int = 100) -> Dict:
        """Get average metrics over a window."""
        window = min(window, len(self.losses))
        if window == 0:
            return {}
        
        return {
            "avg_loss": sum(self.losses[-window:]) / window,
            "avg_tokens_per_sec": sum(self.tokens_per_sec[-window:]) / window,
        }
    
    def reset(self):
        """Reset all metrics."""
        self.losses.clear()
        self.learning_rates.clear()
        self.grad_norms.clear()
        self.tokens_per_sec.clear()
        self.val_losses.clear()
        self.steps.clear()
