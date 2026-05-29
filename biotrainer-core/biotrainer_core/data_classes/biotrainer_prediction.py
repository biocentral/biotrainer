from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union

from ..protocols import Protocol
from ..utils import constants as biotrainer_constants


class BiotrainerPrediction(BaseModel):
    seq_id: str = Field(description="Sequence identifier")
    prediction: Any = Field(description="Predicted value")
    mcd_predictions: Optional[List[Any]] = Field(default=None, description="All Monte-Carlo-Dropout predictions")
    mcd_mean: Optional[Union[float, List[float]]] = Field(default=None, description="Monte-Carlo-Dropout mean(s)")
    mcd_std: Optional[Union[float, List[float]]] = Field(default=None,
                                                         description="Monte-Carlo-Dropout standard deviation(s)")
    mcd_lower_bound: Optional[Union[float, List[float]]] = Field(default=None,
                                                                 description="Monte-Carlo-Dropout lower bound(s)")
    mcd_upper_bound: Optional[Union[float, List[float]]] = Field(default=None,
                                                                 description="Monte-Carlo-Dropout upper bound(s)")
    bald_score: Optional[float] = Field(default=None, description="BALD score")

    def revert_mappings(self, protocol: Protocol,
                        class_int2str: Optional[Dict[int, str]] = None) -> BiotrainerPrediction:
        if class_int2str is None:
            return self

        pred = self.prediction
        mcd_preds = self.mcd_predictions
        if protocol in Protocol.classification_protocols():
            pred = class_int2str[int(pred)]
            mcd_preds = [class_int2str[int(mcd_pred)] for mcd_pred in mcd_preds]

        return BiotrainerPrediction(seq_id=self.seq_id, prediction=pred, mcd_predictions=mcd_preds,
                                    mcd_mean=self.mcd_mean, mcd_std=self.mcd_std,
                                    mcd_lower_bound=self.mcd_lower_bound,
                                    mcd_upper_bound=self.mcd_upper_bound,
                                    bald_score=self.bald_score
                                    )


class BiotrainerResiduePrediction(BiotrainerPrediction):
    residue_index: int = Field(description="Residue index")

    @staticmethod
    def collapse_predictions(predictions: List[BiotrainerResiduePrediction],
                             protocol: Protocol) -> List[BiotrainerPrediction]:
        """ Collapse predictions into a single prediction for each sequence"""
        seq_preds = {}
        for pred in predictions:
            if pred.seq_id not in seq_preds:
                seq_preds[pred.seq_id] = []
            seq_preds[pred.seq_id].append(pred)

        delimiter = biotrainer_constants.RESIDUE_TO_VALUE_TARGET_DELIMITER if protocol in Protocol.regression_protocols() else ""
        return [BiotrainerPrediction(seq_id=seq_id,
                                     prediction=delimiter.join([str(pred.prediction) for pred in
                                                                sorted(preds, key=lambda p: p.residue_index)]))
                for seq_id, preds in seq_preds.items()]
