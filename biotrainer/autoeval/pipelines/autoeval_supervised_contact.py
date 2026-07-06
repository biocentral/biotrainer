import torch
import numpy as np

from pathlib import Path
from typing import Optional, Union, List
from sklearn.linear_model import LogisticRegression

from biotrainer_core.data_classes import SequenceData, ZeroShotContactSingleProtein, ZeroShotContactDatasetResult
from biotrainer_core.input_files import load_contact_map, read_FASTA

from .autoeval_setup import setup_pipeline
from .autoeval_report import ZeroShotContactCachedResults, ZeroShotContactFrameworkReport, AutoEvalReport
from .autoeval_progress import AutoEvalProgress
from ..core import AutoEvalFramework, AutoEvalTask

from ...shared import get_device, Bootstrapper
from ...embedding import get_embedding_service
from ...embedding.huggingface import HuggingfaceTransformerEmbedder
from ...bioengineer.bioengineer_metrics import evaluate_contact_map

# TODO: Logistic Regression Constants
n_train_samples = 20
thershold_c_alpha = 8
l1_penalty = 0.15
seed = 0
min_sep = 6


def _generate_flat_pairwise_dataset_input(seq_data: List[SequenceData], embedding_service, contacts_dir_path):
    x = []
    y = []
    for record in seq_data:
        seq_id = record.seq_id
        sequence = record.seq
        ground_truth_contact_map_path = contacts_dir_path / f"{seq_id}.npy"
        ground_truth_contact_map = load_contact_map(path=ground_truth_contact_map_path,
                                                    sequence=sequence,
                                                    structure_id=seq_id)

        pos = np.arange(ground_truth_contact_map.shape[0])
        diag_idx = np.expand_dims(pos, axis=0) - np.expand_dims(pos, axis=1) >= min_sep

        attention_map = embedding_service._embedder.compute_attention_map(sequence=sequence)

        x.extend(attention_map[diag_idx, :].reshape(-1, attention_map.shape[-1]).to(torch.float32))
        y.extend(ground_truth_contact_map[diag_idx].reshape(-1))

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return x, y


def _generate_per_protein_dataset_input(seq_data: List[SequenceData], embedding_service, contacts_dir_path):
    protein_inputs = []

    for record in seq_data:
        seq_id = record.seq_id
        sequence = record.seq
        ground_truth_contact_map_path = contacts_dir_path / f"{seq_id}.npy"
        ground_truth_contact_map = load_contact_map(path=ground_truth_contact_map_path,
                                                    sequence=sequence,
                                                    structure_id=seq_id)

        attention_map = embedding_service._embedder.compute_attention_map(sequence=sequence)
        protein_inputs.append((seq_id, attention_map, ground_truth_contact_map))

    return protein_inputs


def _run_supervised_contact_tasks(framework: AutoEvalFramework,
                                  embedder_name: str,
                                  autoeval_report: AutoEvalReport,
                                  output_dir: Path,
                                  autoeval_tasks: List[AutoEvalTask],
                                  device=None):
    embedding_service = get_embedding_service(embedder_name=embedder_name, device=get_device(device),
                                              custom_tokenizer_config=None)
    if not isinstance(embedding_service._embedder, HuggingfaceTransformerEmbedder):
        raise ValueError(f"Only HuggingfaceTransformers are supported for supervised contact tasks, "
                         f"but got {embedding_service._embedder}!")
    # TODO Load cached results

    # Execute bioengineer
    # zero_shot_contact_framework_report = ZeroShotContactFrameworkReport.empty(method=zero_shot_method)
    task_names = [task.combined_name() for task in autoeval_tasks]
    print(f"The following tasks will be executed in order: {task_names} (total {len(task_names)})")
    completed_tasks = 0
    total_tasks = len(task_names)
    current_task_name = ""
    for task in autoeval_tasks:
        current_task_name = task.combined_name()
        print(f"Running task {current_task_name}...")
        yield AutoEvalProgress(completed_tasks=completed_tasks,
                               total_tasks=total_tasks,
                               current_task_name=current_task_name,
                               current_framework_name=framework.get_name())
        # TODO: Check if cached result exists for this dataset

        dataset_map = {"train": None,
                       "val": None,
                       "test": []
                       }
        input_dirs = task.input_files
        for input_dir in input_dirs:
            dir_name = input_dir.name
            if dir_name in dataset_map:
                dataset_map[dir_name] = input_dir
            else:
                dataset_map["test"].append(input_dir)

        assert dataset_map["train"] is not None, f"Missing train dataset for task: {current_task_name}"
        assert dataset_map["val"] is not None, f"Missing val dataset for task: {current_task_name}"
        assert len(dataset_map["test"]) > 0, f"Missing test datasets for task: {current_task_name}"

        # Train
        dataset_dir_path = dataset_map["train"]
        fasta_file_path = dataset_dir_path / "extracted_sequences.fasta"
        train_seqs = read_FASTA(fasta_file_path)

        contacts_dir_path = dataset_dir_path / "contacts"
        x_train, y_train = _generate_flat_pairwise_dataset_input(train_seqs, embedding_service, contacts_dir_path)

        # Val (TODO)
        dataset_dir_path = dataset_map["val"]
        fasta_file_path = dataset_dir_path / "extracted_sequences.fasta"
        val_seqs = read_FASTA(fasta_file_path)

        contacts_dir_path = dataset_dir_path / "contacts"
        x_val, y_val = _generate_flat_pairwise_dataset_input(val_seqs, embedding_service, contacts_dir_path)

        # Test
        test_datasets = {}
        for test_path in dataset_map["test"]:
            test_name = test_path.stem
            test_fasta_file_path = test_path / "extracted_sequences.fasta"
            test_seqs = read_FASTA(test_fasta_file_path)
            contacts_dir_path = test_path / "contacts"
            test_datasets[test_name] = _generate_per_protein_dataset_input(test_seqs, embedding_service, contacts_dir_path)
            break  # TODO DEBUG

        # Train classifier on training data
        # Logistic Regression (careful, liblinear does not support int64!)
        clf = LogisticRegression(solver="liblinear", l1_ratio=1, C=l1_penalty)
        clf.fit(x_train, y_train)

        # Test on test sets
        for test_set_name, test_data in test_datasets.items():
            per_protein_results: List[ZeroShotContactSingleProtein] = []
            for protein_name, x_test, ground_truth_contact_map in test_data:
                predicted_contact_map = (clf.predict_proba(x_test.reshape(-1,
                                                                         x_test.shape[-1]))[:, 1].
                                         reshape(ground_truth_contact_map.shape))
                precision_scores = evaluate_contact_map(predicted_contact_map, ground_truth_contact_map)
                single_protein_result = ZeroShotContactSingleProtein(protein_name=protein_name,
                                                                     precision_scores=precision_scores)
                per_protein_results.append(single_protein_result)

            assert len(per_protein_results) > 0, "Empty per-protein results!"

            # Aggregate results
            print(f"Aggregating results for {test_set_name}...")

            seed = 42
            metric_names = list(per_protein_results[0].precision_scores.keys())
            n_proteins = len(per_protein_results)
            values = {metric_name: torch.tensor([[p.precision_scores[metric_name]] for p in per_protein_results],
                                                dtype=torch.float32) for metric_name in metric_names}
            bt_res = Bootstrapper.direct_metrics_bootstrap(metrics=values,
                                                           iterations=30,
                                                           sample_size=n_proteins,
                                                           seed=seed,
                                                           confidence_level=0.05)
            dataset_result = ZeroShotContactDatasetResult(dataset_name=test_set_name,
                                                          aggregated_result=bt_res)
            print(f"Result for {test_set_name}: {dataset_result}")
        # _, dataset_result = bioengineer.evaluate_contact_dataset(dataset_name=current_task_name,
        #                                                         fasta_file_path=fasta_file_path,
        #                                                         contacts_dir_path=contacts_dir_path,
        #                                                         method=zero_shot_method)
        # zero_shot_contact_cached_results.update_and_sync(dataset_name=current_task_name, result=dataset_result,
        #                                                 output_dir=output_dir)
        # zero_shot_contact_framework_report.update_result(task_name=current_task_name, dataset_result=dataset_result)
        completed_tasks += 1
        print(f"Finished task {current_task_name}!")

    # autoeval_report.add_zeroshot_contact_result(framework_name=framework.get_name(),
    #                                            report=zero_shot_contact_framework_report)
    # autoeval_report.write(output_dir=output_dir.parent)
    #
    # print(f"Autoeval pipeline on framework {framework.get_name()} for {embedder_name} finished successfully!")
    yield AutoEvalProgress(completed_tasks=total_tasks, total_tasks=total_tasks,
                           current_task_name=current_task_name,
                           current_framework_name=framework.get_name(),
                           final_report=autoeval_report)


def autoeval_supervised_contact_pipeline(embedder_name: str,
                                         framework: AutoEvalFramework,
                                         autoeval_report: AutoEvalReport,
                                         output_dir: Optional[Union[Path, str]] = "autoeval_output",
                                         force_download: Optional[bool] = False,
                                         custom_storage_path: Optional[Union[Path, str]] = None,
                                         device=None,
                                         ):
    # Setup
    autoeval_tasks = setup_pipeline(data_handler=framework.get_data_handler(),
                                    custom_storage_path=custom_storage_path,
                                    force_download=force_download)
    # Pipeline
    yield from _run_supervised_contact_tasks(framework=framework,
                                             embedder_name=embedder_name,
                                             autoeval_report=autoeval_report,
                                             output_dir=output_dir,
                                             autoeval_tasks=autoeval_tasks,
                                             device=device)
