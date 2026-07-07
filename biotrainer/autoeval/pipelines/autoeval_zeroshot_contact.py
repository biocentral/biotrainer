from pathlib import Path
from typing import Optional, Union, List
from biotrainer_core.data_classes import ZeroShotMethod, ContactSingleProteinResult
from biotrainer_core.data_classes.autoeval import AutoEvalTask, AutoEvalProgress, AutoEvalReport, \
    ContactFrameworkReport, ZeroShotContactCachedResults

from biotrainer_core.input_files import read_FASTA, load_contact_map

from .autoeval_setup import setup_pipeline

from ..core import AutoEvalFramework

from ...shared import get_device
from ...bioengineer import BioEngineer
from ...shared.metrics import evaluate_contact_dataset


def _run_zeroshot_contact_tasks(framework: AutoEvalFramework,
                                embedder_name: str,
                                zero_shot_method: ZeroShotMethod,
                                autoeval_report: AutoEvalReport,
                                output_dir: Path,
                                autoeval_tasks: List[AutoEvalTask],
                                bioengineer: Optional[BioEngineer] = None,
                                device=None):
    if not bioengineer:
        bioengineer = BioEngineer.from_name(name=embedder_name, device=get_device(device))

    assert bioengineer is not None, f"BioEngineer could not be initialized for embedder {embedder_name}!"

    # Load cached results 
    zero_shot_contact_cached_results = ZeroShotContactCachedResults.loaded_or_empty(embedder_name=embedder_name,
                                                                                    method=zero_shot_method,
                                                                                    output_dir=output_dir)
    # Execute bioengineer
    zero_shot_contact_framework_report = ContactFrameworkReport.empty(method=zero_shot_method)
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

        assert len(
            task.input_files) == 1, f"Empty/Multiple input files/dirs not supported for contact tasks! Task: {current_task_name}"

        dataset_dir_path = task.input_files[0]
        fasta_file_path = dataset_dir_path / "extracted_sequences.fasta"
        contacts_dir_path = dataset_dir_path / "contacts"

        # Read fasta file
        seq_records = read_FASTA(fasta_file_path)
        if len(seq_records) == 0:
            raise ValueError(f"Fasta file {fasta_file_path} is empty!")

        # Check for cached results
        cached_results: List[ContactSingleProteinResult] = []
        for seq_record in seq_records:
            seq_id = seq_record.seq_id
            # Check if cached result exists for this dataset
            maybe_cached_result = zero_shot_contact_cached_results.maybe_cached_result(seq_id=seq_id)
            if maybe_cached_result is not None:
                print(f"Skipping dataset {seq_id} as cached result exists for {seq_id}!")
                cached_results.append(maybe_cached_result)

        # Define loading of ground truth contact map
        def load_gt_contact_map(seq_record):
            seq = seq_record.seq
            seq_id = seq_record.seq_id
            ground_truth_contact_map_path = contacts_dir_path / f"{seq_id}.npy"
            return load_contact_map(path=ground_truth_contact_map_path,
                                    sequence=seq,
                                    structure_id=seq_id)

        def evaluate():
            yield from evaluate_contact_dataset(dataset_name=current_task_name,
                                                items=seq_records,
                                                predict_func=lambda
                                                    seq_record: bioengineer.zero_shot_contact_map(
                                                    method=zero_shot_method,
                                                    sequence=seq_record.seq),
                                                get_ground_truth_func=load_gt_contact_map,
                                                get_seq_id_func=lambda d: d.seq_id,
                                                cached_results=cached_results,
                                                )

        for maybe_single_result, maybe_dataset_result in evaluate():
            if maybe_single_result is not None:
                zero_shot_contact_cached_results.update_and_sync(result=maybe_single_result,
                                                                 output_dir=output_dir)
            if maybe_dataset_result is not None:
                zero_shot_contact_framework_report.update_result(task_name=current_task_name,
                                                                 dataset_result=maybe_dataset_result)
                break  # Done

        completed_tasks += 1
        print(f"Finished task {current_task_name}!")

    autoeval_report.add_zeroshot_contact_result(framework_name=framework.get_name(),
                                                report=zero_shot_contact_framework_report)
    autoeval_report.write(output_dir=output_dir.parent)

    print(f"Autoeval pipeline on framework {framework.get_name()} for {embedder_name} finished successfully!")
    yield AutoEvalProgress(completed_tasks=total_tasks, total_tasks=total_tasks,
                           current_task_name=current_task_name,
                           current_framework_name=framework.get_name(),
                           final_report=autoeval_report)


def autoeval_zeroshot_contact_pipeline(embedder_name: str,
                                       framework: AutoEvalFramework,
                                       method: ZeroShotMethod,
                                       autoeval_report: AutoEvalReport,
                                       output_dir: Union[Path, str] = "autoeval_output",
                                       force_download: Optional[bool] = False,
                                       custom_storage_path: Optional[Union[Path, str]] = None,
                                       custom_bioengineer: Optional[BioEngineer] = None,
                                       device=None,
                                       ):
    # Setup
    autoeval_tasks = setup_pipeline(data_handler=framework.get_data_handler(),
                                    custom_storage_path=custom_storage_path,
                                    force_download=force_download)
    # Pipeline
    yield from _run_zeroshot_contact_tasks(framework=framework,
                                           embedder_name=embedder_name,
                                           zero_shot_method=method,
                                           autoeval_report=autoeval_report,
                                           output_dir=Path(output_dir),
                                           autoeval_tasks=autoeval_tasks,
                                           bioengineer=custom_bioengineer,
                                           device=device)
