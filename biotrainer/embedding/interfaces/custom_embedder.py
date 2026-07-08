import torch
from typing import List, Tuple


class CustomEmbedder:
    """ Custom Embedder Interface - Used to provide a standardized interface for autoeval custom embedders """
    def per_residue(self, seqs: List[str]) -> Tuple[str, torch.Tensor]:
        """ Embed a list of sequences and return a tuple of (sequence, per-residue embedding for this sequence) """
        pass

    def per_sequence(self, seqs: List[str]) -> Tuple[str, torch.Tensor]:
        """ Embed a list of sequences and return a tuple of (sequence, per-sequence embedding for this sequence) """
        pass
