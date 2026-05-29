# Unify biocentral prediction_model and biotrainer output
# Add inference functions
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List

from .embedding_stats import EmbeddingStats
from .biotrainer_prediction import BiotrainerPrediction
from .metrics import EpochMetrics, BootstrappedMetric


class DerivedValues(BaseModel):
    """ Derived values calculated during the training process. """
    biotrainer_version: Optional[str] = Field(default=None, description="Version of BioTrainer used for training")
    class_int2str: Optional[Dict[int, str]] = Field(default=None,
                                                    description="Mapping of class integers to class names")
    class_str2int: Optional[Dict[str, int]] = Field(default=None,
                                                    description="Mapping of class names to class integers")
    computed_class_weights: Optional[Dict[int, float]] = Field(default=None,
                                                               description="Class weights computed during training")
    embedding_stats: Optional[EmbeddingStats] = Field(default=None, description="Statistics of the embeddings")
    embeddings_file: Optional[str] = Field(default=None, description="Path to the embeddings file")
    model_hash: Optional[str] = Field(default=None, description="Hash of the model")
    n_classes: Optional[int] = Field(default=None, description="Number of classes in the dataset")
    n_features: Optional[int] = Field(default=None, description="Number of input features (e.g. embedding dimensions)")
    n_testing_ids: Optional[int] = Field(default=None, description="Number of sequences in the test set")
    pipeline_elapsed_time: Optional[float] = Field(default=None, description="Elapsed time in seconds for the pipeline")
    pipeline_end_time: Optional[str] = Field(default=None, description="End time of the pipeline")
    pipeline_start_time: Optional[str] = Field(default=None, description="Start time of the pipeline")
    training_elapsed_time: Optional[float] = Field(default=None, description="Elapsed time in seconds for training")


class SplitSpecificValues(BaseModel):
    """ Values specific to a split. """


class TrainingResult(BaseModel):
    """ Training results for each cross-validation split. """
    n_training_ids: Optional[int] = Field(default=None, description="Number of sequences in the training set")
    n_validation_ids: Optional[int] = Field(default=None, description="Number of sequences in the validation set")
    training_ids: Optional[List[str]] = Field(default=None, description="List of IDs in the training set")
    validation_ids: Optional[List[str]] = Field(default=None, description="List of IDs in the validation set")
    split_hyper_params: Optional[Dict[str, Any]] = Field(default=None,
                                                         description="Hyperparameters used for this split")
    start_time: Optional[str] = Field(default=None, description="Start time of the training process")
    end_time: Optional[str] = Field(default=None, description="End time of the training process")
    elapsed_time: Optional[float] = Field(default=None, description="Elapsed time in seconds for training")
    training_losses: Optional[List[float]] = Field(default=None, description="Training losses for each epoch")
    validation_losses: Optional[List[float]] = Field(default=None, description="Validation losses for each epoch")
    best_epoch_metrics: Optional[EpochMetrics] = Field(default=None, description="Best training epoch metrics")


class TestResult(BaseModel):
    """ Test results after training. """
    metrics: Optional[Dict[str, float]] = Field(default=None, description="Plain test metrics")
    bootstrapped_metrics: List[BootstrappedMetric] = Field(default=None, description="Bootstrapped test metrics")


class BiotrainerModelResult(BaseModel):
    config: Dict[str, Any] = Field(description="Training configuration parameters")
    derived_values: Optional[DerivedValues] = Field(description="Values derived during the training process")
    training_results: Dict[str, TrainingResult] = Field(description="Training results for each cross-validation split")
    test_results: Dict[str, TestResult] = Field(description="Test results after training for each test set")
    predictions: List[BiotrainerPrediction] = Field(description="Predictions made by the model")
