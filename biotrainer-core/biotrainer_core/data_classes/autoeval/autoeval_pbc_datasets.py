from typing import Dict

from enum import Enum
from .autoeval_supervised_dataset import AutoEvalSupervisedDataset

from ..protocol import Protocol


class PBCDatasetName(str, Enum):
    CONSERVATION = "conservation"
    DISORDER_CHEZOD = "disorder_chezod"
    DISORDER_TRIZOD = "disorder_trizod"
    FRUSTRATION_CLASSIFICATION = "frustration-classification"
    FRUSTRATION_REGRESSION = "frustration-regression"
    MEMBRANE = "membrane"
    PHAGES = "phages"
    SCL = "scl"
    SECONDARY_STRUCTURE = "secondary_structure"


def all_pbc_datasets() -> Dict[str, AutoEvalSupervisedDataset]:
    datasets = [AutoEvalSupervisedDataset(name=PBCDatasetName.CONSERVATION,
                                      evaluation_metric="accuracy",
                                      protocol=Protocol.residue_to_class),
            AutoEvalSupervisedDataset(name=PBCDatasetName.DISORDER_CHEZOD,
                                      evaluation_metric="spearmans-corr-coeff",
                                      protocol=Protocol.residue_to_value),
            AutoEvalSupervisedDataset(name=PBCDatasetName.DISORDER_TRIZOD,
                                      evaluation_metric="spearmans-corr-coeff",
                                      protocol=Protocol.residue_to_value),
            AutoEvalSupervisedDataset(name=PBCDatasetName.FRUSTRATION_CLASSIFICATION,
                                      evaluation_metric="macro-f1_score",
                                      protocol=Protocol.residue_to_class),
            AutoEvalSupervisedDataset(name=PBCDatasetName.FRUSTRATION_REGRESSION,
                                      evaluation_metric="spearmans-corr-coeff",
                                      protocol=Protocol.residue_to_value),
            AutoEvalSupervisedDataset(name=PBCDatasetName.MEMBRANE,
                                      evaluation_metric="macro-f1_score",
                                      protocol=Protocol.residue_to_class),
            AutoEvalSupervisedDataset(name=PBCDatasetName.PHAGES,
                                      evaluation_metric="accuracy",
                                      protocol=Protocol.sequence_to_class),
            AutoEvalSupervisedDataset(name=PBCDatasetName.SCL,
                                      evaluation_metric="accuracy",
                                      protocol=Protocol.sequence_to_class),
            AutoEvalSupervisedDataset(name=PBCDatasetName.SECONDARY_STRUCTURE,
                                      evaluation_metric="accuracy",
                                      protocol=Protocol.residue_to_class),
            ]
    return {d.name: d for d in datasets}

