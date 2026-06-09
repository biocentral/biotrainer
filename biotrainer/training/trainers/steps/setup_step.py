import time
import datetime

from pathlib import Path
from junban import PipelineStep

from biotrainer_core.data_classes import Protocol
from biotrainer_core.functions.seeding import seed_all
from biotrainer_core.functions.hashing import calculate_model_hash

from ..hp_manager import HyperParameterManager
from ..pipeline_context import BiotrainerPipelineContext

from ....shared import get_logger, __version__, setup_logging, get_device

logger = get_logger(__name__)


class SetupStep(PipelineStep[BiotrainerPipelineContext]):

    def _check_entry_assumptions(self, context: BiotrainerPipelineContext) -> bool:
        return True

    def _check_exit_assumptions(self, context: BiotrainerPipelineContext) -> bool:
        assert context.model_hash is not None, f"model_hash cannot be None after the setup step!"
        return True

    def get_start_message(self) -> str:
        return "Running setup..."

    def get_end_message(self) -> str:
        return "Setup complete!"

    @staticmethod
    def _post_process_config(context: BiotrainerPipelineContext):
        context.config["protocol"] = Protocol.from_string(context.config["protocol"])

        # Create output dir
        output_dir = Path(context.config["output_dir"])
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True)
        context.config["output_dir"] = output_dir

        # Create log directory (if necessary)
        log_dir = output_dir / context.model_hash
        if not log_dir.is_dir():
            log_dir.mkdir(parents=True)
        context.config["log_dir"] = str(log_dir)

        # Setup logging
        setup_logging(str(log_dir), context.config["num_epochs"])
        logger.info(f"Logging training at: {log_dir}")

        # Get device once at the beginning
        device = get_device(context.config["device"] if "device" in context.config.keys() else None)
        context.config["device"] = device

    def _execute(self, context: BiotrainerPipelineContext) -> BiotrainerPipelineContext:
        context.pipeline_start_time = time.perf_counter()
        pipeline_start_time_abs = str(datetime.datetime.now().isoformat())

        # Calculate model hash
        model_hash = calculate_model_hash(config=context.config,
                                          input_data=context.input_data,
                                          custom_trainer=context.custom_pipeline
                                          )
        context.model_hash = model_hash

        self._post_process_config(context)

        # Log version
        logger.info(f"** Running biotrainer (v{__version__}) training routine **")
        context.output_manager.update_derived_values(biotrainer_version=str(__version__))
        # Log start time
        logger.info(f"Pipeline start time: {pipeline_start_time_abs}")
        context.output_manager.update_derived_values(pipeline_start_time=pipeline_start_time_abs)

        if "pretrained_model" in context.config.keys():
            logger.info(f"Using pre_trained model: {context.config['pretrained_model']}")

        # Create hyperparameter manager
        hp_manager = HyperParameterManager(**context.config)
        context.hp_manager = hp_manager

        logger.info(f"Training {context.config['model_choice']} model with hash: {model_hash}")
        context.output_manager.update_derived_values(model_hash=model_hash)
        # Seed
        seed = context.config["seed"]
        seed_all(seed)
        logger.info(f"Using seed: {seed}")
        # Log device
        logger.info(f"Using device: {context.config['device']}")

        context.output_manager.add_config(context.config)
        return context
