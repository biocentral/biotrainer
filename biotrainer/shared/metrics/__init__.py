from biotrainer_core.data_classes import Protocol

from .ndcg import NDCG
from .metrics_calculator import MetricsCalculator, ResidueClassificationMetricsCalculator, \
    ResidueRegressionMetricsCalculator, ResiduesClassificationMetricsCalculator, ResiduesRegressionMetricsCalculator, \
    SequenceClassificationMetricsCalculator, SequenceRegressionMetricsCalculator
from .ci_bounds import get_mean_and_confidence_bounds

METRIC_CALCULATORS = {
    Protocol.residue_to_class: ResidueClassificationMetricsCalculator,
    Protocol.residue_to_value: ResidueRegressionMetricsCalculator,
    Protocol.residues_to_class: ResiduesClassificationMetricsCalculator,
    Protocol.residues_to_value: ResiduesRegressionMetricsCalculator,
    Protocol.sequence_to_class: SequenceClassificationMetricsCalculator,
    Protocol.sequence_to_value: SequenceRegressionMetricsCalculator,
}

__all__ = [
    "METRIC_CALCULATORS",
    "NDCG",
    "MetricsCalculator",
    "get_mean_and_confidence_bounds",
]
