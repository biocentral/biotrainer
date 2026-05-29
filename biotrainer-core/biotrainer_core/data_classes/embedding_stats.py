from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


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
