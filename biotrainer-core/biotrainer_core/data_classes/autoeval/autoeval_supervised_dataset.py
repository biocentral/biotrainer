from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

from ..protocol import Protocol


class AutoEvalSupervisedDataset(BaseModel):
    name: str = Field(description="Name of the dataset")
    evaluation_metric: str = Field(description="Evaluation metric")
    protocol: Protocol = Field(description="Protocol to use for evaluation")
    splits: Optional[List[str]] = Field(default=None, description="Splits as part of the dataset")
