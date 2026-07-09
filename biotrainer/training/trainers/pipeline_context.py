from pathlib import Path
from junban import PipelineContext
from typing import Dict, Any, Union, List, Optional
from biotrainer_core.data_classes import SequenceData, Protocol

from .target_manager import TargetManager

from ..output_files import OutputManager

from ...embedding import PeftEmbeddingService


class BiotrainerPipelineContext(PipelineContext):
    """Context object that maintains state throughout the pipeline execution"""

    def __init__(self, config: Dict[str, Any], output_manager: OutputManager, custom_pipeline: bool):
        # Values set prior to pipeline execution
        self.config = config
        self.protocol: Protocol = Protocol.from_string(config["protocol"])  # Convenience access to protocol config
        self.output_manager = output_manager
        self.custom_pipeline = custom_pipeline

        # Data produced during pipeline execution
        # Setup
        self.pipeline_start_time = None
        self.model_hash = None
        self.hp_manager = None
        self.skip_signal = False  # If True, existing model result has been loaded so all subsequent steps are skipped
        # Input Data
        self.input_data: Optional[Union[Path, List[SequenceData]]] = None
        self.hash2id = None  # Dict to Map from sequence hash to sequence id
        # Embedding + Projection
        self.id2emb = None
        self.embedding_service: Optional[PeftEmbeddingService] = None  # For fine-tuning only
        # Data Loading
        self.target_manager: Optional[TargetManager] = None
        self.n_features = None  # Embedding shape
        self.n_classes = None
        self.class_str2int: Optional[Dict[str, int]] = None  # Used to apply random_masking
        self.train_dataset = None
        self.val_dataset = None
        self.test_datasets = None
        self.baseline_test_datasets = None  # For random model baseline, uses non-finetuned embeddings
        self.prediction_dataset = None
        self.class_weights = None

        # Training
        self.best_split = None

        # Timing information
        self.pipeline_end_time = None
        self.step_timings = {}
