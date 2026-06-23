"""Logger for wandb integration."""
from typing import Dict, Optional
import wandb


class WandBLogger:
    """Weights & Biases logger for training."""
    
    def __init__(
        self,
        project: str = "Lite-GPT",
        entity: Optional[str] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
    ):
        """Initialize wandb logger.
        
        Args:
            project: WandB project name
            entity: WandB entity (team/username)
            config: Configuration dictionary to log
            name: Run name
        """
        self.project = project
        self.entity = entity
        self.run = None
        self.config = config or {}
        self.name = name
        self._enabled = True
    
    def start(self):
        """Start a new wandb run."""
        try:
            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                config=self.config,
                name=self.name,
            )
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            self._enabled = False
    
    def log(self, metrics: Dict, step: Optional[int] = None):
        """Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Global step number
        """
        if not self._enabled or self.run is None:
            return
        
        try:
            if step is not None:
                self.run.log(metrics, step=step)
            else:
                self.run.log(metrics)
        except Exception as e:
            print(f"Warning: Failed to log to wandb: {e}")
    
    def finish(self):
        """Finish the wandb run."""
        if self.run is not None:
            try:
                self.run.finish()
            except Exception as e:
                print(f"Warning: Failed to finish wandb run: {e}")
    
    def is_enabled(self) -> bool:
        """Check if wandb is enabled."""
        return self._enabled and self.run is not None
