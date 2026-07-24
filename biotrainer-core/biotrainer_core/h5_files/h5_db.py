""" Module for treating h5 files as databases """
import h5py

from pathlib import Path
from datetime import datetime
from typing import Optional, Set, Generator, Union, List

from .emb_db_dto import EmbeddingDatabaseDTO

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    np = None
    _NUMPY_AVAILABLE = False


def read_h5_db(h5_file_path: Union[str, Path], ids_to_load: Optional[Set[str]] = None) -> Generator[
    EmbeddingDatabaseDTO, None, None]:
    """ Read embedding database DTOs from h5 file """
    with h5py.File(h5_file_path, "r") as h5_file:
        for ds_name in h5_file.keys():
            ds = h5_file[ds_name]

            # If ids_to_load is specified, check if this sequence hash should be loaded
            sequence_hash = ds.attrs["sequence_hash"]
            if ids_to_load is not None and sequence_hash not in ids_to_load:
                continue

            # Load the embedding data
            embedding_data = ds[:]

            # Determine if this was per_residue or per_sequence based on the 'reduced' flag
            is_reduced = ds.attrs["reduced"]

            if is_reduced:
                embd_per_sequence = embedding_data
                embd_per_residue = None
            else:
                embd_per_sequence = None
                embd_per_residue = embedding_data

            # Create the DTO from the dataset and its attributes
            embd_dto = EmbeddingDatabaseDTO(
                hash_key=sequence_hash,
                seq_len=int(ds.attrs["sequence_length"]),
                access_count=int(ds.attrs["access_count"]),
                created_at=datetime.fromisoformat(ds.attrs["created_at"]),
                last_accessed=datetime.fromisoformat(ds.attrs["last_accessed"]),
                embedder_name=ds.attrs["embedder_name"],
                embd_per_sequence=embd_per_sequence,
                embd_per_residue=embd_per_residue,
                keep=bool(ds.attrs["keep"])
            )

            yield embd_dto


def write_h5_db(h5_file_path: Union[str, Path], embd_db_dtos: List[EmbeddingDatabaseDTO]) -> Generator[int, None, None]:
    """ Write embedding database DTOs to an h5 file. Returns the number of written embeddings as a generator function"""
    with h5py.File(h5_file_path, "a") as h5_file:
        progress = 0
        yield progress
        for embd_dto in embd_db_dtos:
            # We save per_residue if it exists, else per_sequence (per_sequence can be easily deducted)
            data = embd_dto.embd_per_residue if embd_dto.embd_per_residue is not None else embd_dto.embd_per_sequence

            if data is None:
                raise ValueError(f"No embedding data found for sequence hash: {embd_dto.hash_key}!")

            if _TORCH_AVAILABLE:
                if torch.is_tensor(data):
                    data = data.cpu().numpy()

            # Key must be unique in H5. If multiple models for same sequence, use hash_model
            ds_name = f"{embd_dto.hash_key}_{embd_dto.cleaned_embedder_name()}"
            ds = h5_file.create_dataset(ds_name, data=data, compression="gzip")

            # Save all metadata as attributes
            ds.attrs["sequence_hash"] = embd_dto.hash_key
            ds.attrs["sequence_length"] = embd_dto.seq_len
            ds.attrs["access_count"] = embd_dto.access_count
            ds.attrs["created_at"] = embd_dto.created_at.isoformat()
            ds.attrs["last_accessed"] = embd_dto.last_accessed.isoformat()
            ds.attrs["embedder_name"] = embd_dto.embedder_name
            ds.attrs["keep"] = embd_dto.keep

            # Flag if it was per_sequence or per_residue
            ds.attrs["reduced"] = embd_dto.reduced()
            progress += 1
            yield progress
