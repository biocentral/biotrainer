from abc import ABC, abstractmethod
from biotrainer_core.data_classes import BiotrainerModelUpdate


class BiotrainerOutputObserver(ABC):
    @abstractmethod
    def update(self, data: BiotrainerModelUpdate) -> None:
        """Handle an output event with associated data."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Cleanup resources when training is finished."""
        pass
