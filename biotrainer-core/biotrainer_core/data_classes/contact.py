from pydantic import BaseModel, Field
from typing import Dict, List, Optional

from .metrics import BootstrappedMetric


class ContactSingleProteinResult(BaseModel):
    """Required for bootstrapping and for caching and resuming from mid-dataset!"""
    protein_name: str = Field(description="Name of the protein")
    precision_scores: Dict[str, float] = Field(
        description="Precision scores for the protein's predicted contacts")  # e.g. {"short_P@L2": 0.83, "short_P@L5": 0.78, "long_P@L2": 0.81, "long_P@L5": 0.76}


class ContactDatasetResult(BaseModel):
    dataset_name: str = Field(description="Name of the dataset")
    aggregated_result: List[BootstrappedMetric] = Field(description="Aggregated and bootstrapped "
                                                                    "precision scores for the dataset")

    def __str__(self) -> str:
        scores = ", ".join(f"{metric.name}: {metric.mean:.3f}" for metric in self.aggregated_result)
        if not scores:
            return f"Dataset result [{self.dataset_name}] - No scores available"
        return f"Dataset result [{self.dataset_name}] - {scores}"

    def long_PatL2(self) -> Optional[float]:
        long_p_at_l2 = [metric for metric in self.aggregated_result if metric.name == "long_P@L2"]
        if len(long_p_at_l2) == 0:
            return None
        return long_p_at_l2[0].mean
