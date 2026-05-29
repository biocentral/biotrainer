import os
import shutil
import tempfile

from pathlib import Path
from typing import Union, Dict, Any, Optional, List
from biotrainer_core.input_files import read_FASTA
from biotrainer_core.data_classes import BiotrainerModelResult, SequenceData

from .inference import Inferencer
from .trainers.pipeline import Pipeline
from .output_files import BiotrainerOutputObserver, InferenceOutputManager
from .utilities.executer import parse_config_file_and_execute_run

from ..embedders import get_embedding_service


class BiotrainerModel:
    def __init__(self, training_result: Optional[BiotrainerModelResult] = None, ):
        self.training_result = training_result
        self._inferencer: Optional[Inferencer] = None
        self._iom: Optional[InferenceOutputManager] = None

    @classmethod
    def from_training_result(cls, training_result: Union[BiotrainerModelResult, Path, str]):
        if isinstance(training_result, BiotrainerModelResult):
            return cls(training_result=training_result)
        elif isinstance(training_result, (Path, str)):
            with open(training_result, "r") as f:
                tr_res = f.read()
            return cls(training_result=BiotrainerModelResult.model_validate_json(tr_res))
        else:
            raise TypeError(f"Invalid type for training_result: {type(training_result)}")

    def train(self, config: Union[str, Path, Dict[str, Any]],
              custom_pipeline: Optional[Pipeline] = None,
              custom_output_observers: Optional[List[BiotrainerOutputObserver]] = None) -> BiotrainerModelResult:
        if self.training_result is not None:
            print(f"Warning: Training result already available! Overwriting with new training result..")

        training_result = parse_config_file_and_execute_run(config=config, custom_pipeline=custom_pipeline,
                                                            custom_output_observers=custom_output_observers)
        self.training_result = training_result
        return training_result

    def inferencer(self):
        if self._inferencer is not None:
            return self.inferencer
        if self.training_result is None:
            raise ValueError("No training result available!")
        inferencer, iom = Inferencer.create_from_out_file(out_file_path=self.training_result,  # TODO
                                                          automatic_path_correction=True)
        self._inferencer = inferencer
        self._iom = iom
        return inferencer

    def inference_output_manager(self):
        if self._iom is not None:
            return self._iom
        _ = self.inferencer()
        return self._iom

    def _predict_from_records(self,
                              seq_data: List[SequenceData],
                              save_embeddings: Optional[bool] = False,
                              scale_embeddings: Optional[bool] = True,
                              ):
        input_ids = {record.get_id_for_id2emb(): record.seq_id for record in seq_data}
        inferencer = self.inferencer()
        iom = self.inference_output_manager()

        embedding_service = get_embedding_service(embedder_name=iom.embedder_name(),
                                                  custom_tokenizer_config=None,  # TODO
                                                  use_half_precision=iom.use_half_precision(),
                                                  device=iom.device())
        adapter_path = iom.adapter_path()
        if adapter_path is not None:
            embedding_service.add_finetuned_adapter(adapter_path=adapter_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            result_file = embedding_service.compute_embeddings(input_data=seq_data,
                                                               output_dir=Path(tmpdir),
                                                               protocol=iom.protocol(),
                                                               )
            embeddings = embedding_service.load_embeddings(result_file)

            if save_embeddings:
                shutil.copy(result_file, os.getcwd())

        result = inferencer.from_embeddings(embeddings=embeddings, scale_embeddings=scale_embeddings)[
            "mapped_predictions"]

        sorted_results = []
        for seq_hash, prediction in result.items():
            input_id = input_ids[seq_hash]
            sorted_results.append((input_id, seq_hash, prediction))

        sorted_results = sorted(sorted_results, key=lambda x: x[0])

        for input_id, seq_hash, prediction in sorted_results:
            print(f"Prediction for {input_id} (sequence hash {seq_hash}):\n\t{prediction}")

        return result

    def predict(self,
                model_input: Union[str, List[SequenceData]],
                save_embeddings: Optional[bool] = False,
                scale_embeddings: Optional[bool] = True,
                ) -> Dict[str, Any]:
        """ Convenience function to create predictions from a fasta file or sequence data list.
            Use the .inferencer() method to get a more flexible and powerful interface.
        """
        if self.training_result is None:
            raise ValueError("No training result available!")

        if isinstance(model_input, list) and isinstance(model_input[0], SequenceData):
            sequence_data = model_input
        elif isinstance(model_input, str):
            if "." in model_input and Path(model_input).exists():
                sequence_data = read_FASTA(model_input)
            else:
                model_input_split = [seq for seq in model_input.split(",")]
                sequence_data = [SequenceData(seq_id=f"Seq{idx}",
                                              seq=seq) for idx, seq in enumerate(model_input_split)]
        else:
            raise ValueError("model_input must be a Path to an input file or a comma separated list of sequences!")

        return self._predict_from_records(
            seq_data=sequence_data,
            save_embeddings=save_embeddings,
            scale_embeddings=scale_embeddings
        )
