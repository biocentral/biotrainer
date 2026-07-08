from __future__ import annotations

import torch

from pathlib import Path
from datetime import datetime
from biotrainer_core.data_classes import ZeroShotMethod
from typing import Optional, Callable, Tuple, List, Union, Iterable, Generator
from biotrainer_core.data_classes.autoeval import AutoEvalProgress, AutoEvalReport

from .core import AutoEvalFramework, AutoEvalMode
from .pipelines import (setup_output_dir, validate_input, autoeval_supervised_pipeline,
                        autoeval_zeroshot_pipeline, autoeval_zeroshot_contact_pipeline,
                        autoeval_supervised_contact_pipeline, setup_pipeline)
from .pipelines.autoeval_supervised import get_unique_framework_sequences, check_h5_file, setup_embedder, \
    CustomEmbedder, run_supervised_pipeline  # TODO Move
from .autoeval_frameworks import AvailableFramework

from ..bioengineer import BioEngineer
from ..training.output_files import BiotrainerOutputObserver


class AutoEval:
    def __init__(self,
                 embedder_name: str,
                 output_dir: Union[Path, str] = "autoeval_output",
                 force_download: bool = False,
                 use_half_precision: bool = False,
                 min_seq_length: int = 0,
                 max_seq_length: int = 2000,
                 custom_storage_path: Optional[Union[Path, str]] = None,
                 precomputed_per_residue_embeddings: Optional[Path] = None,
                 precomputed_per_sequence_embeddings: Optional[Path] = None,
                 custom_embedder: Optional[CustomEmbedder] = None,
                 custom_bioengineer: Optional[BioEngineer] = None
                 ):
        if force_download and custom_storage_path:
            raise ValueError(f"Cannot force download and use custom storage path at the same time!"
                             f"force_download only clears the cache directory, "
                             f"so it is not necessary when using custom_storage_path, "
                             f"just make sure that that is up-to-date.")
        if (precomputed_per_residue_embeddings is None) ^ (precomputed_per_sequence_embeddings is None):
            raise ValueError(
                f"You must provide either paths to both precomputed per-sequence and per-residue embeddings "
                f"or no precomputed path at all!")
        using_precomputed_embeddings = precomputed_per_residue_embeddings is not None and precomputed_per_sequence_embeddings is not None

        using_custom_embedding_functions = custom_embedder is not None

        if using_precomputed_embeddings and using_custom_embedding_functions:
            raise ValueError(f"You must either provide precomputed embeddings or custom embedding functions, not both!")

        self.output_dir = Path(output_dir)  # TODO SETUP?
        self.embedder_name = embedder_name
        self.force_download = force_download
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.use_half_precision = use_half_precision
        self.custom_storage_path = custom_storage_path

        # Embeddings
        self.precomputed_per_residue_embeddings = precomputed_per_residue_embeddings
        self.precomputed_per_sequence_embeddings = precomputed_per_sequence_embeddings
        self.custom_embedder = custom_embedder

        # Bioengineer
        self.bioengineer = custom_bioengineer

        self._tasks = []
        self._results = {}

    def _setup_supervised_pipeline(self):
        all_unique_per_res = {}
        all_unique_per_seq = {}
        for framework_obj in self._tasks:
            task_config_tuples, unique_per_res, unique_per_seq = get_unique_framework_sequences(framework=framework_obj,
                                                                                                min_seq_length=self.min_seq_length,
                                                                                                max_seq_length=self.max_seq_length,
                                                                                                custom_storage_path=self.custom_storage_path,
                                                                                                force_download=self.force_download)
            all_unique_per_res.update(unique_per_res)
            all_unique_per_seq.update(unique_per_seq)

    def _pre_embed(self, all_unique_per_residue, all_unique_per_sequence, ):
        embedder = setup_embedder(embedder_name=self.embedder_name,
                                  output_dir=self.output_dir,
                                  precomputed_per_residue_embeddings=self.precomputed_per_residue_embeddings,
                                  precomputed_per_sequence_embeddings=self.precomputed_per_sequence_embeddings,
                                  custom_embedder=self.custom_embedder,
                                  )
        # Embed (TODO: Parallelize)
        print(f"Embedding {len(all_unique_per_residue)} sequences per_residue")
        embeddings_file_per_residue = embedder.per_residue_path(
            [seq_record.seq for _, seq_record in all_unique_per_residue.items()]
        )

        print(f"Embedding {len(all_unique_per_sequence)} sequences per_sequence")
        embeddings_file_per_sequence = embedder.per_sequence_path(
            [seq_record.seq for _, seq_record in all_unique_per_sequence.items()]
        )

        check_h5_file(name="per-residue", h5_path=embeddings_file_per_residue,
                      expected_length=len(all_unique_per_residue))
        check_h5_file(name="per-sequence", h5_path=embeddings_file_per_sequence,
                      expected_length=len(all_unique_per_sequence))

        print("Calculated embeddings successfully!")

    def _general_task_setup(self, available_framework: AvailableFramework) -> Optional:
        framework_obj: AutoEvalFramework = validate_input(available_framework,
                                                          zero_shot_method=None,
                                                          min_seq_length=self.min_seq_length,
                                                          max_seq_length=self.max_seq_length)

        # Setup
        output_dir = setup_output_dir(base_dir=self.output_dir,
                                      embedder_name=self.embedder_name,
                                      framework_name=framework_obj.get_name())
        # Check if results already exist
        autoeval_report = AutoEvalReport.loaded_or_empty(embedder_name=self.embedder_name,
                                                         training_date=str(datetime.now().date().isoformat()),
                                                         output_dir=output_dir.parent)
        # Framework results already exist -> skip execution
        maybe_framework_result = autoeval_report.maybe_framework_result(framework_name=framework_obj.get_name())
        if maybe_framework_result:
            print(f"Autoeval report for framework {available_framework} already exists, "
                  f"execution will be skipped!")

            self._results[framework_obj.get_name()] = maybe_framework_result

        return framework_obj, autoeval_report, maybe_framework_result

    def _supervised_task(self, available_framework: AvailableFramework,
                         custom_output_observers: List[BiotrainerOutputObserver] = None, ):
        framework_obj, autoeval_report, maybe_framework_result = self._general_task_setup(available_framework)
        if maybe_framework_result:
            return self

        self._tasks.append(
            lambda task_config_tuples, path_per_res, path_per_seq, device: run_supervised_pipeline(
                embedder_name=self.embedder_name,
                framework=framework_obj,
                autoeval_report=autoeval_report,
                embeddings_file_per_residue=path_per_res,
                embeddings_file_per_sequence=path_per_seq,
                output_dir=self.output_dir,
                task_config_tuples=task_config_tuples,
                min_seq_length=self.min_seq_length,
                max_seq_length=self.max_seq_length,
                custom_output_observers=custom_output_observers,
                device=device)
        )
        return self

    def pbc_supervised(self,
                       custom_output_observers: List[BiotrainerOutputObserver] = None,
                       ) -> AutoEval:
        return self._supervised_task(AvailableFramework.PBC_SUPERVISED, custom_output_observers)

    def flip(self,
             custom_output_observers: List[BiotrainerOutputObserver] = None,
             ) -> AutoEval:
        return self._supervised_task(AvailableFramework.FLIP, custom_output_observers)

    def pgym(self, zero_shot_method: ZeroShotMethod):
        framework_obj, autoeval_report, maybe_framework_result = self._general_task_setup(AvailableFramework.PGYM)
        if maybe_framework_result:
            return self

        self._tasks.append(
            lambda task_config_tuples, path_per_res, path_per_seq, bioengineer, device: autoeval_zeroshot_pipeline(
                framework=framework_obj,
                embedder_name=self.embedder_name,
                autoeval_tasks=task_config_tuples,
                zero_shot_method=zero_shot_method,
                autoeval_report=autoeval_report,
                output_dir=self.output_dir,
                bioengineer=bioengineer,
                device=device)
        )
        return self

    def pbc_zeroshot_contact(self, zero_shot_method: ZeroShotMethod = ZeroShotMethod.JACOBIAN_CONTACT):
        framework_obj, autoeval_report, maybe_framework_result = self._general_task_setup(AvailableFramework.PBC_ZEROSHOT_CONTACT)
        if maybe_framework_result:
            return self
        self._tasks.append(
            lambda task_config_tuples, path_per_res, path_per_seq, bioengineer,
                   device: autoeval_zeroshot_contact_pipeline(framework=framework_obj,
                                                              embedder_name=self.embedder_name,
                                                              zero_shot_method=zero_shot_method,
                                                              autoeval_report=autoeval_report,
                                                              autoeval_tasks=task_config_tuples,
                                                              output_dir=self.output_dir,
                                                              bioengineer=bioengineer,
                                                              )
        )
        return self

    def pbc_supervised_contact(self):
        framework_obj, autoeval_report, maybe_framework_result = self._general_task_setup(AvailableFramework.PBC_SUPERVISED_CONTACT)
        if maybe_framework_result:
            return self
        self._tasks.append(
            lambda task_config_tuples, path_per_res, path_per_seq, bioengineer, device: autoeval_supervised_contact_pipeline(
                framework=framework_obj,
                embedder_name=self.embedder_name,
                autoeval_tasks=task_config_tuples,
                autoeval_report=autoeval_report,
                output_dir=self.output_dir,
                device=device)
        )
        return self


    def run(self, device: Optional[Union[str, torch.device]] = None) -> AutoEvalReport:
        for task in self._tasks:
            current_progress = None
            for progress in task(device):
                print(progress)
                current_progress = progress
            if current_progress is None:
                raise RuntimeError("No progress was returned from autoeval task!")
            final_report = current_progress.final_report
            if final_report is None:
                raise RuntimeError("No final report was returned from autoeval task!")
            self._results[final_report.framework_name] = final_report

        # TODO Construct autoeval report
        # TODO Check if report is still necessary for pipeline functions
        # TODO Task Runner
        # TODO Return AutoEval Report

    def run_parallel(self, devices: Optional[List[Union[str, torch.device]]] = None) -> AutoEvalReport:
        # TODO Parallelization
        return self.run(device=devices[0] if devices else None)


def autoeval_pipeline(embedder_name: str,
                      framework: Union[str, AvailableFramework],
                      zero_shot_method: Optional[ZeroShotMethod] = None,
                      output_dir: Optional[Union[Path, str]] = "autoeval_output",
                      force_download: Optional[bool] = False,
                      use_half_precision: Optional[bool] = False,
                      min_seq_length: Optional[int] = 0,
                      max_seq_length: Optional[int] = 2000,
                      custom_tokenizer_config: Optional[dict] = None,
                      precomputed_per_residue_embeddings: Optional[Path] = None,
                      precomputed_per_sequence_embeddings: Optional[Path] = None,
                      custom_embedding_function_per_residue: Optional[
                          Callable[[Iterable[str]], Generator[Tuple[str, torch.Tensor], None, None]]] = None,
                      custom_embedding_function_per_sequence: Optional[
                          Callable[[Iterable[str]], Generator[Tuple[str, torch.Tensor], None, None]]] = None,
                      custom_storage_path: Optional[Union[Path, str]] = None,
                      custom_output_observers: List[BiotrainerOutputObserver] = None,
                      custom_bioengineer: Optional[BioEngineer] = None,
                      device: Optional[Union[str, torch.device]] = None,
                      ) -> Generator[AutoEvalProgress, None, None]:
    """
    Run the autoeval pipeline for a given embedder_name and framework.

    :param embedder_name: The name of the embedder. Usually a huggingface pretrained embedder in format org/embed_name.
    :param framework: The framework to be evaluated. Currently, only FLIP is available.
    :param zero_shot_method: The zero-shot method to use. Only for zero-shot framework evaluation.
    :param output_dir: The directory to save the output to, defaults to "autoeval_output".
    :param force_download: Flag to determine whether to force re-downloading the framework datasets, defaults to False.
    :param use_half_precision: Flag to determine whether to use a half-precision floating point for the embedder or not.
    :param min_seq_length: The minimum sequence length to pre-filter the framework datasets, defaults to 0.
    :param max_seq_length: The maximum sequence length to pre-filter the framework datasets, defaults to 2000.
    :param custom_pipeline: Custom pipeline to be executed, defaults to None (= default biotrainer pipeline).
        If a custom pipeline is specified, no other custom parameters for embedding must be provided. The pipeline
        must handle embeddings on its own.
    :param custom_tokenizer_config: Custom tokenizer configuration dictionary for onnx models.
    :param precomputed_per_residue_embeddings:
        Optional path to precomputed per-residue embeddings.
        Must be provided together with per-sequence embeddings path.
        The embeddings must be stored by sequence hash in a .h5 file. Lear more here: docs/h5_file_standardization.md
    :param precomputed_per_sequence_embeddings:
        Optional path to precomputed per-sequence embeddings.
        Must be provided together with per-residue embeddings path.
        The embeddings must be stored by sequence hash in a .h5 file. Lear more here: docs/h5_file_standardization.md
    :param custom_embedding_function_per_residue:
        Custom per-residue embedding function that is used instead
        of the biotrainer embedding service if provided.
        Takes an iterable of sequence strings as input and must provide the per-residue embeddings as a generator.
    :param custom_embedding_function_per_sequence:
        Custom per-sequence embedding function that is used instead
        of the biotrainer embedding service if provided.
        Takes an iterable of sequence strings as input and must provide the per-sequence embeddings as a generator.
    :param custom_storage_path: Optional path where to store the framework datasets if not downloaded yet.
    :param custom_output_observers: Optional list of custom training output observers.
    :param custom_bioengineer: Optional custom bioengineer instance to use for zero-shot evaluation.
    :param device: Optional device specifier for embedding/model computations (e.g., 'cuda:0', 'cuda:1', 'cpu').
    :return: A dictionary containing the autoeval pipeline results. Each task result is a biotrainer model output dict.
    """
    framework_obj: AutoEvalFramework = validate_input(framework,
                                                      zero_shot_method=zero_shot_method,
                                                      min_seq_length=min_seq_length,
                                                      max_seq_length=max_seq_length)

    if force_download and custom_storage_path:
        raise ValueError(f"Cannot force download and use custom storage path at the same time!"
                         f"force_download only clears the cache directory, "
                         f"so it is not necessary when using custom_storage_path, "
                         f"just make sure that that is up-to-date.")

    # Setup
    output_dir = setup_output_dir(base_dir=output_dir,
                                  embedder_name=embedder_name,
                                  framework_name=framework_obj.get_name())
    # Check if results already exist
    autoeval_report = AutoEvalReport.loaded_or_empty(embedder_name=embedder_name,
                                                     training_date=str(datetime.now().date().isoformat()),
                                                     output_dir=output_dir.parent)
    # Framework results already exist -> skip execution
    maybe_framework_result = autoeval_report.maybe_framework_result(framework_name=framework_obj.get_name())
    if maybe_framework_result:
        print(f"Autoeval report for framework {framework_obj.get_name()} already exists, skipping execution!")
        yield AutoEvalProgress(completed_tasks=maybe_framework_result.number_tasks(),
                               total_tasks=maybe_framework_result.number_tasks(),
                               current_task_name="",
                               current_framework_name=framework_obj.get_name(),
                               final_report=autoeval_report)
        return

    # Framework results do not exist yet -> execute autoeval pipeline
    match framework_obj.get_mode():
        case AutoEvalMode.SUPERVISED:
            yield from autoeval_supervised_pipeline(embedder_name=embedder_name,
                                                    framework=framework_obj,
                                                    autoeval_report=autoeval_report,
                                                    output_dir=output_dir,
                                                    force_download=force_download,
                                                    use_half_precision=use_half_precision,
                                                    min_seq_length=min_seq_length,
                                                    max_seq_length=max_seq_length,
                                                    custom_tokenizer_config=custom_tokenizer_config,
                                                    precomputed_per_residue_embeddings=precomputed_per_residue_embeddings,
                                                    precomputed_per_sequence_embeddings=precomputed_per_sequence_embeddings,
                                                    custom_embedding_function_per_residue=custom_embedding_function_per_residue,
                                                    custom_embedding_function_per_sequence=custom_embedding_function_per_sequence,
                                                    custom_storage_path=custom_storage_path,
                                                    custom_output_observers=custom_output_observers,
                                                    device=device)
        case AutoEvalMode.ZERO_SHOT:
            yield from autoeval_zeroshot_pipeline(embedder_name=embedder_name,
                                                  framework=framework_obj,
                                                  method=zero_shot_method,
                                                  autoeval_report=autoeval_report,
                                                  output_dir=output_dir,
                                                  force_download=force_download,
                                                  custom_storage_path=custom_storage_path,
                                                  custom_bioengineer=custom_bioengineer,
                                                  device=device)
        case AutoEvalMode.ZERO_SHOT_CONTACT:
            yield from autoeval_zeroshot_contact_pipeline(embedder_name=embedder_name,
                                                          framework=framework_obj,
                                                          method=zero_shot_method,
                                                          autoeval_report=autoeval_report,
                                                          output_dir=output_dir,
                                                          force_download=force_download,
                                                          custom_storage_path=custom_storage_path,
                                                          custom_bioengineer=custom_bioengineer,
                                                          device=device)

        case AutoEvalMode.SUPERVISED_CONTACT_ATTENTION:
            yield from autoeval_supervised_contact_pipeline(embedder_name=embedder_name,
                                                            framework=framework_obj,
                                                            autoeval_report=autoeval_report,
                                                            output_dir=output_dir,
                                                            force_download=force_download,
                                                            custom_storage_path=custom_storage_path,
                                                            device=device,
                                                            )
