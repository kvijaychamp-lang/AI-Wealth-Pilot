"""Training orchestration pipeline module."""
from __future__ import annotations

from dataclasses import dataclass

from training.trainer import TrainingPipeline as Trainer
from config.settings import AppConfig


@dataclass
class TrainingPipeline:
    """High-level interface for executing the training pipeline."""
    config: AppConfig

    def run(self) -> None:
        trainer = Trainer(self.config)
        trainer.run()
