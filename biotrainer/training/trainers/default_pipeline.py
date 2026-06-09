from junban import Pipeline
from typing import Any, Dict

from .pipeline_context import BiotrainerPipelineContext
from .steps import SetupStep, ScalingStep, EmbeddingStep, FineTuningEmbeddingStep, ProjectionStep, DataLoadingStep, \
    TrainingStep, TestingStep, PostProcessStep, InputValidationStep, DatasetCreationStep

from ...shared import get_logger

logger = get_logger(__name__)


class DefaultPipeline:
    def __init__(self, config: Dict[str, Any]):
        if "finetuning_config" in config:
            self.pipeline = self._default_finetuning_pipeline()
        else:
            self.pipeline = self._default_pipeline()

    @staticmethod
    def _default_pipeline() -> Pipeline[BiotrainerPipelineContext]:
        return Pipeline(name="TRAINING PIPELINE",
                        steps=[
                            InputValidationStep(),
                            SetupStep(),
                            EmbeddingStep(),
                            DataLoadingStep(),
                            ScalingStep(),
                            ProjectionStep(),
                            DatasetCreationStep(),
                            TrainingStep(),
                            TestingStep(),
                            PostProcessStep(),
                        ],
                        logger=logger,
                        )

    @staticmethod
    def _default_finetuning_pipeline() -> Pipeline[BiotrainerPipelineContext]:
        return Pipeline(name="FINETUNING PIPELINE",
                        steps=[
                            SetupStep(),
                            InputValidationStep(),
                            FineTuningEmbeddingStep(),
                            DataLoadingStep(),
                            DatasetCreationStep(),
                            TrainingStep(),
                            TestingStep(),
                            PostProcessStep()
                        ],
                        logger=logger,
                        )
