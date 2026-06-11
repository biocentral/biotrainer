from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, Callable


# Not using pydantic for faster transfer
@dataclass(frozen=True)
class EmbeddingDatabaseDTO:
    hash_key: str
    seq_len: int
    access_count: int
    created_at: datetime
    last_accessed: datetime
    embedder_name: str
    embd_per_sequence: Optional  # Can be numpy array or torch tensor or bytes (compressed)
    embd_per_residue: Optional  # Can be numpy array or torch tensor or bytes (compressed)
    keep: bool  # Keep this embedding in the database and do not remove it via cleanup

    def reduced(self) -> bool:
        return self.embd_per_sequence is not None and self.embd_per_residue is None

    def to_tuple(self) -> Tuple[str, int, int, datetime, datetime, str, Optional, Optional, bool]:
        return (self.hash_key,
                self.seq_len,
                self.access_count,
                self.created_at,
                self.last_accessed,
                self.embedder_name,
                self.embd_per_sequence,
                self.embd_per_residue,
                self.keep)

    def cleaned_embedder_name(self) -> str:
        return self.embedder_name.replace('/', '_')

    def compressed(self, compression_strategy: Callable) -> EmbeddingDatabaseDTO:
        per_seq_compressed = compression_strategy(self.embd_per_sequence)
        per_res_compressed = compression_strategy(self.embd_per_residue)
        return EmbeddingDatabaseDTO(
            hash_key=self.hash_key,
            seq_len=self.seq_len,
            access_count=self.access_count,
            created_at=self.created_at,
            last_accessed=self.last_accessed,
            embedder_name=self.embedder_name,
            embd_per_sequence=per_seq_compressed,
            embd_per_residue=per_res_compressed,
            keep=self.keep,
        )

    def decompressed(self, decompression_strategy: Callable) -> EmbeddingDatabaseDTO:
        per_seq_decompressed = decompression_strategy(self.embd_per_sequence)
        per_res_decompressed = decompression_strategy(self.embd_per_residue)
        return EmbeddingDatabaseDTO(
            hash_key=self.hash_key,
            seq_len=self.seq_len,
            access_count=self.access_count,
            created_at=self.created_at,
            last_accessed=self.last_accessed,
            embedder_name=self.embedder_name,
            embd_per_sequence=per_seq_decompressed,
            embd_per_residue=per_res_decompressed,
            keep=self.keep,
        )