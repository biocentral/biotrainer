import h5py

from pathlib import Path
from typing import Optional, Set, Dict, Any, Union


def read_id2emb(h5_file_path: Union[str, Path], ids_to_load: Optional[Set[str]] = None) -> Dict[str, Any]:
    """ Read raw content from h5_file_path and return as a dictionary """
    # Old version see:
    # https://stackoverflow.com/questions/48385256/optimal-hdf5-dataset-chunk-shape-for-reading-rows/48405220#48405220
    # Sequence hash from embeddings file -> Embedding
    id2emb = {}
    with h5py.File(h5_file_path, 'r') as embeddings_file:
        if ids_to_load is None:
            # Load all sequences
            id2emb = {idx: embedding[:] for (idx, embedding) in embeddings_file.items()}
        else:
            for idx in ids_to_load:
                if idx in embeddings_file:
                    id2emb[idx] = embeddings_file[idx][:]
                else:
                    raise ValueError(f"Sequence hash {idx} not found in embeddings file!")
    return id2emb
