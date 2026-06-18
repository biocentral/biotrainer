import pandas as pd

from typing import Union, Optional
from pathlib import Path

from .fasta import write_FASTA

from ..data_classes import SequenceData


def pgym_csv_to_fasta(pgym_csv_path: Union[str, Path],
                      output_fasta_path: Union[str, Path],
                      single_mutations_only: Optional[bool] = False,
                      use_binary_score: Optional[bool] = False,
                      ) -> int:
    # Read ProteinGym dataset
    df = pd.read_csv(pgym_csv_path)
    if len(df) == 0:
        raise ValueError(f"Dataset file {pgym_csv_path} is empty!")

    try:
        score_column = "DMS_score_bin" if use_binary_score else "DMS_score"

        mutation_fitness = {row["mutated_sequence"]: (row["mutant"], row[score_column]) for _, row in df.iterrows()}
        if single_mutations_only:
            mutation_fitness = {seq: (mut, score) for seq, (mut, score) in mutation_fitness.items() if ":" not in mut}
    except KeyError as e:
        raise ValueError(f"Dataset file {pgym_csv_path} is missing a required column: {e}")

    sequence_data = [SequenceData(seq_id=f"mut_{mut}", seq=seq, label=str(score))
                     for seq, (mut, score) in mutation_fitness.items()]

    return write_FASTA(output_fasta_path, sequence_data)
