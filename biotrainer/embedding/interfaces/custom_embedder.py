import torch

from abc import ABC, abstractmethod
from typing import List, Tuple, Generator


class CustomEmbedder(ABC):
    """ Custom Embedder Interface - Used to provide a standardized interface for autoeval custom embedders """

    @abstractmethod
    def per_residue(self, sequences: List[str]) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """ Embed a list of sequences and yield a tuple of (sequence, per-residue embedding for this sequence) """
        pass

    @abstractmethod
    def per_sequence(self, sequences: List[str]) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """ Embed a list of sequences and yield a tuple of (sequence, per-sequence embedding for this sequence) """
        pass
