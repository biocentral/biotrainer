import torch
import numpy as np

from typing import Iterable
from pydantic import BaseModel


class EmbeddingStats(BaseModel):
    embedder_name: str
    dims: int
    n_tracked: int
    max: float
    min: float


class EmbeddingStatsTracker:
    def __init__(self, embedder_name: str):
        self._embedder_name = embedder_name

        self.max = float("-inf")
        self.min = float("inf")
        self.dims = None
        self.n_tracked = 0

    def track_entire_dataset(self, per_residue_embeddings: Iterable):
        for emb in per_residue_embeddings:
            if isinstance(emb, torch.Tensor):
                emb = emb.numpy()
            if isinstance(emb, list):
                emb = np.array(emb)
            self.track(emb)

    def track(self, per_residue_embedding: np.ndarray):
        self.n_tracked += per_residue_embedding.shape[0]
        if self.dims is None:
            self.dims = per_residue_embedding.shape[1]
        self.max = max(self.max, per_residue_embedding.max())
        self.min = min(self.min, per_residue_embedding.min())

    def get_stats(self) -> EmbeddingStats:
        return EmbeddingStats(embedder_name=self._embedder_name,
                              dims=self.dims,
                              n_tracked=self.n_tracked,
                              max=self.max,
                              min=self.min)
