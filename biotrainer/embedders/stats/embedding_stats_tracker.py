from __future__ import annotations

import torch
import numpy as np

from pydantic import BaseModel
from typing import Iterable, Optional


class EmbeddingStats(BaseModel):
    embedder_name: str
    dims: int
    n_tracked: int
    min: float
    max: float

    @staticmethod
    def from_biotrainer_result(biotrainer_result: dict) -> Optional[EmbeddingStats]:
        embd_stats = biotrainer_result["derived_values"].get("embedding_stats")
        if embd_stats is None:
            return None
        return EmbeddingStats.model_validate(embd_stats)

    def accumulate_results(self, other: Optional[EmbeddingStats]):
        if other is None:
            return self
        if self.embedder_name != other.embedder_name:
            raise ValueError(
                f"Inconsistent embedder name in embedding stats: {self.embedder_name} vs {other.embedder_name}")
        if self.dims != other.dims:
            raise ValueError(f"Inconsistent dimensions in embedding stats: {self.dims} vs {other.dims}")
        self.n_tracked += other.n_tracked
        self.min = min(self.min, other.min)
        self.max = max(self.max, other.max)
        return self


class EmbeddingStatsTracker:
    """ Tracks statistics about embeddings (currently only per-residue embeddings)."""

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
        self.min = min(self.min, per_residue_embedding.min())
        self.max = max(self.max, per_residue_embedding.max())

    def get_stats(self) -> EmbeddingStats:
        return EmbeddingStats(embedder_name=self._embedder_name,
                              dims=self.dims,
                              n_tracked=self.n_tracked,
                              min=self.min,
                              max=self.max)
