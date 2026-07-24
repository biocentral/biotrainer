from typing import Dict
from enum import Enum

from .autoeval_supervised_dataset import AutoEvalSupervisedDataset
from ..protocol import Protocol


class FLIPDatasetName(str, Enum):
    AAV = "aav"
    BIND = "bind"
    CONSERVATION = "conservation"
    GB1 = "gb1"
    MELTOME = "meltome"
    SCL = "scl"
    SAV = "sav"
    SECONDARY_STRUCTURE = "secondary_structure"


def all_flip_datasets() -> Dict[str, AutoEvalSupervisedDataset]:
    datasets = [AutoEvalSupervisedDataset(
        name=f"{FLIPDatasetName.AAV}",
        evaluation_metric="spearmans-corr-coeff",
        protocol=Protocol.sequence_to_value,
        splits=["des_mut", "mut_des", "one_vs_many", "two_vs_many", "seven_vs_many", "low_vs_high", "sampled"]
    ), AutoEvalSupervisedDataset(
        name=f"{FLIPDatasetName.BIND}",
        evaluation_metric="macro-f1_score",
        protocol=Protocol.residue_to_class,
        splits=["one_vs_many", "two_vs_many", "from_publication", "one_vs_sm", "one_vs_mn", "one_vs_sn"]
    ), AutoEvalSupervisedDataset(
        name=f"{FLIPDatasetName.CONSERVATION}/sampled",
        evaluation_metric="accuracy",
        protocol=Protocol.residue_to_class
    ), AutoEvalSupervisedDataset(
        name=f"{FLIPDatasetName.GB1}",
        evaluation_metric="spearmans-corr-coeff",
        protocol=Protocol.sequence_to_value,
        splits=["one_vs_rest", "two_vs_rest", "three_vs_rest", "low_vs_high", "sampled"]
    ), AutoEvalSupervisedDataset(
        name=f"{FLIPDatasetName.MELTOME}",
        evaluation_metric="spearmans-corr-coeff",
        protocol=Protocol.sequence_to_value,
        splits=["mixed_split", "human", "human_cell"]
    ), AutoEvalSupervisedDataset(
        name=f"{FLIPDatasetName.SCL}",
        evaluation_metric="accuracy",
        protocol=Protocol.sequence_to_class,
        splits=["mixed_soft", "mixed_hard", "human_soft", "human_hard", "balanced", "mixed_vs_human_2"]
    ), AutoEvalSupervisedDataset(
        name=f"{FLIPDatasetName.SAV}",
        evaluation_metric="f1_score",
        protocol=Protocol.sequence_to_class,
        splits=["mixed", "human", "only_savs"]
    ), AutoEvalSupervisedDataset(
        name=f"{FLIPDatasetName.SECONDARY_STRUCTURE}",
        evaluation_metric="accuracy",
        protocol=Protocol.residue_to_class,
        splits=["sampled"]
    )]

    return {d.name: d for d in datasets}
