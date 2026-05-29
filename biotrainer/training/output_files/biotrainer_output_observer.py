from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Tuple
from biotrainer_core.data_classes import EpochMetrics, BiotrainerModelResult


class OutputData(BaseModel):
    current_model_result: BiotrainerModelResult = Field(description="Current model result")
    training_iteration: Optional[Tuple[str, EpochMetrics]] = Field(default=None,
                                                                   description="Current training iteration for fast "
                                                                               "updates of observers like tensorboard")


class BiotrainerOutputObserver(ABC):
    @abstractmethod
    def update(self, data: OutputData) -> None:
        """Handle an output event with associated data."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Cleanup resources when training is finished."""
        pass
