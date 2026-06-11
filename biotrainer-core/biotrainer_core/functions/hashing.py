import h5py
import json
import hashlib

from pathlib import Path
from copy import deepcopy
from typing import Dict, Any, List, Optional


def calculate_sequence_hash(sequence: str) -> str:
    suffix = len(sequence)
    sequence = f"{sequence}_{suffix}"
    return hashlib.sha256(sequence.encode()).hexdigest()


def hash_sequence_data(input_data: List) -> str:
    input_data = sorted(input_data, key=lambda x: x.get_hash())
    entries = [
        {
            "record": x.model_dump_json(),       # sequence + attributes (no embedding)
            "embedding_hash": x.get_embedding_hash(),  # cheap SHA256 of raw bytes
        }
        for x in input_data
    ]
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True).encode()
    ).hexdigest()


def hash_h5_file(file_path: Optional[Path]) -> str:
    """Load all keys from the h5 file and hash them."""
    if file_path is None or not file_path.exists():
        return ""

    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())
    return hashlib.sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest()


def calculate_model_hash(
        config: Dict[Any, Any],
        input_data: List,
        custom_trainer: bool,
) -> str:
    """
    Create a deterministic hash representing dataset files and model configuration.

    Args:
        config: Dictionary containing model configuration
        input_data: List of SequenceData objects
        custom_trainer: If true, a custom trainer is used

    Returns:
        A hex string hash uniquely identifying this model setup
    """
    config = deepcopy(config)
    # 0. Remove input_file from config (is represented by input_data)
    config.pop('input_file', None)
    config.pop('input_data', None)
    embeddings_file = config.pop('embeddings_file', None)

    # 1. Calculate embedding file hash
    embeddings_file_hash = hash_h5_file(embeddings_file)

    # 2. Calculate sequence data hash
    sequence_data_hash = hash_sequence_data(input_data)

    # 3. Prepare config hash (normalize it to ensure consistency)
    # Sort keys and convert everything to string
    config_normalized = json.dumps({str(k): str(v) for k, v in config.items()}, sort_keys=True)
    config_hash = hashlib.sha256(config_normalized.encode()).hexdigest()

    # 4. Combine all hashes
    combined = {
        'embeddings_file_hash': embeddings_file_hash,
        'config_hash': config_hash,
        'sequence_data_hash': sequence_data_hash,
        'custom_trainer': str(custom_trainer)
    }

    # 5. Create final hash
    final_hash = hashlib.sha256(json.dumps(combined, sort_keys=True).encode()).hexdigest()

    return final_hash[:16]  # First 16 chars is sufficient for us
