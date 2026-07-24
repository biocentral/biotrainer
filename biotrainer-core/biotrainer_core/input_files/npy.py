import numpy as np

from pathlib import Path
from typing import Union, Optional


def load_contact_map(path: Union[str, Path],
                     sequence: Optional[str] = None,
                     structure_id: Optional[str] = None) -> np.ndarray:

    path = Path(path)
    if not path.exists():
        raise ValueError(f"Contact map {path} does not exist!")

    contact_map = np.load(path)

    # Validate
    if structure_id is None:
        structure_id = path.stem

    if contact_map.size == 0:
        raise ValueError(f"Empty contact map for {structure_id}!")

    if sequence is not None:
        seq_len = len(sequence)
        if contact_map.shape != (seq_len, seq_len):
            raise ValueError(
                f"Shape mismatch for {structure_id}: expected ({seq_len}, {seq_len}), got {contact_map.shape}!")

    return contact_map