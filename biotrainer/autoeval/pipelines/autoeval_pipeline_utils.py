import random

from typing import List
from biotrainer_core.data_classes import SequenceData


def subsample_seq_records_for_contact_development_mode(seq_records: List[SequenceData]) -> List[SequenceData]:
    initial_n = len(seq_records)
    sample_ratio_casp = 0.4  # Higher ratio for the rather small datasets to keep a meaningful sample
    sample_ratio_selected = 0.05
    if initial_n < 100:
        sample_ratio = sample_ratio_casp
    else:
        sample_ratio = sample_ratio_selected
    rng = random.Random(14)
    sample = rng.sample(seq_records, int(len(seq_records) * sample_ratio))

    print(f"Subsampled {initial_n} sequences to {len(sample)} for contact development mode.")
    return sample