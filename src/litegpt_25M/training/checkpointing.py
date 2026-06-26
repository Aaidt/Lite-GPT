"""Checkpoint management for training."""

import json
import shutil
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from safetensors.torch import (
    save_model,
    load_model,
)

class CheckpointManager:
    """Manage model checkpoints."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, Any],
        is_best: bool = False,
    ):
        """Save a checkpoint.

        Args:
            step: Training step number
            model: Model to save
            optimizer: Optimizer to save
            metrics: Metrics to save with checkpoint
            is_best: Whether this is the best checkpoint so far
        """
        ckpt_dir = self.checkpoint_dir / f"step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights as safetensors
        save_model(model, str(ckpt_dir / "model.safetensors"))

        # Save training state (optimizer, step, metrics) as torch
        training_state = {
            "step": step,
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }
        torch.save(training_state, ckpt_dir / "training_state.pt")

        metadata = {
            "step": step,
            "metrics": {
                k: float(v) if isinstance(v, (int, float))
                else str(v)
                for k, v in metrics.items()
            },
        }
        # Save best checkpoint (overwrites previous best)
        if is_best:
            best_dir = self.checkpoint_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            save_model(model, str(best_dir / "model.safetensors"))
            torch.save(training_state, best_dir / "training_state.pt")
            with open(best_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

        # Save metadata
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def load(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load a checkpoint.

        Args:
            model: Model to load into
            optimizer: Optimizer to load into
            checkpoint_path: Path to checkpoint directory, model.safetensors,
                             or legacy .pt file. Defaults to best checkpoint.

        Returns:
            Checkpoint data including step and metrics
        """
        if checkpoint_path is None:
            best_dir = self.checkpoint_dir / "best"
            if best_dir.exists():
                return self._load_from_dir(best_dir, model, optimizer)
            # Fall back to old format
            old_best = self.checkpoint_dir / "checkpoint_best.pt"
            if old_best.exists():
                return self._load_old_format(model, optimizer, old_best)
            print("Warning: No checkpoint found")
            return {}

        cp = Path(checkpoint_path)
        if cp.is_dir():
            return self._load_from_dir(cp, model, optimizer)
        if cp.suffix == ".safetensors":
            return self._load_from_dir(cp.parent, model, optimizer)
        return self._load_old_format(model, optimizer, cp)

    def _load_from_dir(self, ckpt_dir: Path, model, optimizer) -> Dict[str, Any]:
        model_file = ckpt_dir / "model.safetensors"
        training_file = ckpt_dir / "training_state.pt"

        if not model_file.exists():
            print(f"Warning: Model weights not found at {model_file}")
            return {}

        load_model(model, str(model_file))

        if training_file.exists():
            training_state = torch.load(training_file, weights_only=True)
            optimizer.load_state_dict(training_state["optimizer_state_dict"])
            return {
                "step": training_state["step"],
                "metrics": training_state["metrics"],
            }

        print(f"Warning: Training state not found at {training_file}")
        return {"step": 0, "metrics": {}}

    def _load_old_format(self, model, optimizer, path: Path) -> Dict[str, Any]:
        checkpoint = torch.load(path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return {
            "step": checkpoint["step"],
            "metrics": checkpoint["metrics"],
        }

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, Any]:
        """Load the latest checkpoint.

        Args:
            model: Model to load into
            optimizer: Optimizer to load into

        Returns:
            Checkpoint data including step and metrics
        """
        step_dirs = sorted(
            [
                d
                for d in self.checkpoint_dir.iterdir()
                if d.is_dir() and d.name.startswith("step_")
            ],
            key=lambda d: int(d.name.split("_")[1]),
        )
        if step_dirs:
            return self._load_from_dir(step_dirs[-1], model, optimizer)

        # Fall back to old format
        old = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        if old:
            return self._load_old_format(model, optimizer, old[-1])

        print("No checkpoints found")
        return {}

    def get_checkpoint_list(self) -> list:
        """Get list of all checkpoint directories, sorted by step."""
        step_dirs = [
            d
            for d in self.checkpoint_dir.iterdir()
            if d.is_dir() and d.name.startswith("step_")
        ]
        return sorted(step_dirs, key=lambda d: int(d.name.split("_")[1]))

    def delete_old_checkpoints(self, keep_last: int = 3):
        """Delete old checkpoints, keeping only the last N.

        Args:
            keep_last: Number of recent checkpoints to keep
        """
        checkpoints = self.get_checkpoint_list()
        if len(checkpoints) > keep_last:
            for ckpt_dir in checkpoints[:-keep_last]:
                shutil.rmtree(ckpt_dir)