import torch

from ..interfaces import BaselineEmbedder


class LengthEmbedder(BaselineEmbedder):
    """
    Baseline embedder: Calculate sequence length. Returns a Lx1 dimension embedding for per-residue and
    a 1x1 dimension embedding for per-sequence.

    This embedder is meant to be used as a naive baseline to compare against other pretrained embedders.
    """

    embedding_dimension = 1
    name = "length_embedder"

    def _embed_single(self, sequence: str) -> torch.Tensor:
        return torch.ones(len(sequence), 1, dtype=torch.float32)

    @staticmethod
    def reduce_per_protein(embedding: torch.Tensor) -> torch.Tensor:
        return torch.tensor([embedding.size(0)], dtype=torch.float32)