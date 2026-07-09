from junban import PipelineStep

from ..pipeline_context import BiotrainerPipelineContext

from ..target_manager import TargetManager

from ....shared import get_logger

logger = get_logger(__name__)


class DataLoadingStep(PipelineStep[BiotrainerPipelineContext]):

    def _check_entry_assumptions(self, context: BiotrainerPipelineContext) -> bool:
        id2emb = context.id2emb
        assert id2emb is not None and len(id2emb) > 0, f"id2emb cannot be None or empty at the data loading step!"
        return True

    def _check_exit_assumptions(self, context: BiotrainerPipelineContext) -> bool:
        assert context.target_manager is not None, f"target_manager cannot be None after the data loading step!"
        return True

    def get_start_message(self) -> str:
        return "Loading data..."

    def get_end_message(self) -> str:
        return "Data loaded!"

    def _skip(self, context: BiotrainerPipelineContext) -> bool:
        return context.skip_signal

    def _execute(self, context: BiotrainerPipelineContext) -> BiotrainerPipelineContext:
        # Load TARGETS and SETS from input data
        target_manager = TargetManager(protocol=context.config["protocol"],
                                       input_data=context.input_data,
                                       ignore_file_inconsistencies=context.config["ignore_file_inconsistencies"],
                                       cross_validation_method=context.config["cross_validation_config"]["method"],
                                       interaction=context.config.get("interaction"))
        target_manager.load()

        context.target_manager = target_manager
        return context
