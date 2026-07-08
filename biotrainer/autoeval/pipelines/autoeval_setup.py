import os

from pathlib import Path
from typing import List, Optional, Union
from biotrainer_core.data_classes import ZeroShotMethod
from biotrainer_core.data_classes.autoeval import AutoEvalTask, AutoEvalMode

from ..autoeval_frameworks import framework_factory
from ..core import AutoEvalDataHandler, AutoEvalFramework


def setup_output_dir(base_dir: Path, embedder_name: str, framework_name: str) -> Path:
    embedder_dir_name = embedder_name
    if "/" in embedder_dir_name:  # Huggingface
        embedder_dir_name = embedder_dir_name.replace("/", "-")

    output_dir = Path(base_dir) / embedder_dir_name / framework_name

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir


def validate_input(framework,
                   zero_shot_method: Optional[ZeroShotMethod],
                   min_seq_length: Optional[int],
                   max_seq_length: Optional[int]) -> AutoEvalFramework:
    framework_obj = framework_factory(framework)

    if framework_obj is None:
        raise ValueError(f"Unsupported framework: {framework}")

    match framework_obj.get_mode():
        case AutoEvalMode.SUPERVISED:  # Supervised frameworks
            if zero_shot_method is not None:
                raise ValueError("Zero-shot method must not be provided for a supervised framework!")
            if min_seq_length is None or max_seq_length is None:
                raise ValueError("min_seq_length and max_seq_length must be provided for a supervised framework!")
            if min_seq_length >= max_seq_length:
                raise ValueError("min_seq_length must be less than max_seq_length")
            if max_seq_length <= 0:
                raise ValueError("max_seq_length must be greater than 0")
        case AutoEvalMode.ZERO_SHOT:  # Zero-Shot frameworks
            if zero_shot_method is None:
                raise ValueError("Zero-shot method must be provided for a zero-shot framework!")
            if zero_shot_method == ZeroShotMethod.JACOBIAN_CONTACT:
                raise ValueError(
                    "Zero-shot method JACOBIAN_CONTACT currently only supported in mode ZERO_SHOT_CONTACT!")
        case AutoEvalMode.ZERO_SHOT_CONTACT:  # Zero-Shot contact frameworks
            if zero_shot_method is None:
                raise ValueError("Zero-shot method must be provided for a zero-shot framework!")
            if zero_shot_method != ZeroShotMethod.JACOBIAN_CONTACT:
                raise ValueError(
                    "Only zero-shot method JACOBIAN_CONTACT currently supported in mode ZERO_SHOT_CONTACT!")

    return framework_obj


def setup_pipeline(data_handler: AutoEvalDataHandler,
                   min_seq_length: Optional[int] = None,
                   max_seq_length: Optional[int] = None,
                   custom_storage_path: Optional[Union[Path, str]] = None,
                   force_download: Optional[bool] = False,
                   ) -> List[AutoEvalTask]:
    framework_base_path = data_handler.get_framework_base_path(
        custom_storage_path=custom_storage_path)

    if force_download:
        data_handler.clear_autoeval_cache()

    if not os.path.exists(framework_base_path):
        os.makedirs(framework_base_path, exist_ok=True)

    if data_handler.is_download_necessary(framework_base_path):
        data_handler.download_data(data_dir=framework_base_path)
    data_handler.preprocess(base_path=framework_base_path,
                            min_seq_length=min_seq_length,
                            max_seq_length=max_seq_length)
    auto_eval_tasks = data_handler.get_tasks(base_path=framework_base_path,
                                             min_seq_length=min_seq_length,
                                             max_seq_length=max_seq_length)

    return auto_eval_tasks
