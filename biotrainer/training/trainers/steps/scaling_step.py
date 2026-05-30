from pathlib import Path

from junban import PipelineStep

from ..pipeline_context import BiotrainerPipelineContext

from ...utilities import FeatureScaler
from ....shared import get_logger

logger = get_logger(__name__)


class ScalingStep(PipelineStep[BiotrainerPipelineContext]):

    def _check_entry_assumptions(self, context: BiotrainerPipelineContext) -> bool:
        id2emb = context.id2emb
        assert id2emb is not None and len(id2emb) > 0, f"id2emb cannot be None or empty at the scaling step!"
        assert context.target_manager is not None, f"target_manager cannot be None at the scaling step!"
        return True

    def _check_exit_assumptions(self, context: BiotrainerPipelineContext) -> bool:
        return True

    def get_start_message(self) -> str:
        return "Scaling features..."

    def get_end_message(self) -> str:
        return "Feature scaling complete!"

    def _execute(self, context: BiotrainerPipelineContext) -> BiotrainerPipelineContext:
        id2emb = context.id2emb
        target_manager = context.target_manager
        old_n_embeddings = len(id2emb)

        scaling_method = context.config.get("scaling_method", "none")
        if scaling_method != "none":
            training_ids = set(target_manager.training_ids)
            training_embs = {seq_id: embd for seq_id, embd in id2emb.items() if seq_id in training_ids}
            other_embs = {seq_id: embd for seq_id, embd in id2emb.items() if seq_id not in training_ids}

            # Fit on training embeddings
            feature_scaler = FeatureScaler(method=scaling_method, protocol=context.config["protocol"])
            feature_scaler = feature_scaler.fit(training_embs, context.target_manager._id2target)

            # Transform all embeddings
            training_embs_scaled = feature_scaler.transform(training_embs)
            other_embs_scaled = feature_scaler.transform(other_embs)
            id2emb = {**training_embs_scaled, **other_embs_scaled}

            # Save fitted scaler
            save_dir = context.config["log_dir"]
            scaling_save_name = feature_scaler.get_file_name(feature_scaler.method)
            feature_scaler.save(Path(save_dir) / scaling_save_name)
            logger.info(f"Fitted feature scaling {scaling_method}!")
        else:
            logger.info(f"No feature scaling performed (as configured).")

        assert old_n_embeddings == len(id2emb), f"The number of embeddings changed during feature scaling!"
        context.id2emb = id2emb

        return context
