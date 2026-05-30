import pickle

from pathlib import Path
from typing import Dict, Any
from biotrainer_core.data_classes import Protocol

from junban import PipelineStep

from ..pipeline_context import BiotrainerPipelineContext

from ....shared import get_logger
from ....embedding import EmbeddingService

logger = get_logger(__name__)


class ProjectionStep(PipelineStep[BiotrainerPipelineContext]):

    def _check_entry_assumptions(self, context: BiotrainerPipelineContext) -> bool:
        id2emb = context.id2emb
        assert id2emb is not None and len(id2emb) > 0, f"id2emb cannot be None or empty at the projection step!"
        assert context.target_manager is not None, f"target_manager cannot be None at the projection step!"
        return True

    def _check_exit_assumptions(self, context: BiotrainerPipelineContext) -> bool:
        return True

    def get_start_message(self) -> str:
        return "Running projection..."

    def get_end_message(self) -> str:
        return "Projection complete!"

    @staticmethod
    def _is_dimension_reduction_possible(context: BiotrainerPipelineContext, dimension_reduction_method, n_reduced_components,
                                         id2emb: Dict[str, Any]) -> bool:
        protocol: Protocol = context.config["protocol"]

        min_number_embeddings = 3
        min_number_dimensions = 3

        number_embeddings = len(id2emb)
        number_dimensions = next(iter(id2emb.values())).shape[0]
        if (protocol.using_per_sequence_embeddings() and dimension_reduction_method and n_reduced_components and
                number_embeddings >= min_number_embeddings and
                number_dimensions >= min_number_dimensions):
            return True
        else:
            if dimension_reduction_method and n_reduced_components:
                # Check for errors
                if number_embeddings < min_number_embeddings:
                    raise ValueError(f"Dimensionality reduction cannot be performed as \
                                the number of samples is less than {min_number_embeddings}")
                if number_dimensions < 3:
                    raise ValueError(f"Dimensionality reduction cannot be performed as \
                                the original embedding dimension is less than {min_number_dimensions}")
                if not protocol.using_per_sequence_embeddings():
                    raise ValueError(f"Dimensionality reduction cannot be performed as \
                                the embeddings are not per-sequence embeddings")
            return False

    def _execute(self, context: BiotrainerPipelineContext) -> BiotrainerPipelineContext:
        id2emb = context.id2emb
        target_manager = context.target_manager
        old_n_embeddings = len(id2emb)

        dimension_reduction_method = context.config.get("dimension_reduction_method")
        n_reduced_components = context.config.get("n_reduced_components")

        if self._is_dimension_reduction_possible(context, dimension_reduction_method, n_reduced_components, id2emb):
            training_ids = set(target_manager.training_ids)
            training_embs = {seq_id: embd for seq_id, embd in id2emb.items() if seq_id in training_ids}
            other_embs = {seq_id: embd for seq_id, embd in id2emb.items() if seq_id not in training_ids}
            training_embs_reduced, fitted_transform = EmbeddingService.embeddings_dimensionality_reduction(
                embeddings=training_embs,
                dimension_reduction_method=dimension_reduction_method,
                n_reduced_components=n_reduced_components)
            other_embs_reduced, _ = EmbeddingService.embeddings_dimensionality_reduction(
                embeddings=other_embs,
                dimension_reduction_method=dimension_reduction_method,
                n_reduced_components=n_reduced_components,
                fitted_transform=fitted_transform
            )
            # Combine embeddings
            id2emb = {**training_embs_reduced, **other_embs_reduced}
            # Save fitted transform
            save_dir = context.config["log_dir"]
            transform_save_name = f"{dimension_reduction_method}_{n_reduced_components}_transform.pkl"
            with open(Path(save_dir) / transform_save_name, "wb") as f:
                pickle.dump(fitted_transform, f)
            logger.info(f"Fitted dimensionality reduction {dimension_reduction_method} with {n_reduced_components}!")
        else:
            logger.info(f"No dimension reduction performed (as configured).")

        assert old_n_embeddings == len(id2emb), f"The number of embeddings changed during dimensionality reduction!"
        context.id2emb = id2emb

        return context
