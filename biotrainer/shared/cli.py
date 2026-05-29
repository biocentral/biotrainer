import cyclopts

from pathlib import Path
from typing import Union, Dict, Any, Optional, List
from biotrainer_core.data_classes import BiotrainerModelResult, Protocol

from ..embedding import EmbeddingAPI
from ..training import BiotrainerModel
from ..autoeval import autoeval_pipeline

app = cyclopts.App()


@app.command
def train(config: Union[str, Path, Dict[str, Any]]) -> BiotrainerModelResult:
    """
       Entry point for training

       @param config: Biotrainer configuration file path or config dict
       """
    return BiotrainerModel().train(config)


@app.command
def predict(training_output_file: Union[str, Path],
            model_input: str,
            save_embeddings: Optional[bool] = False,
            scale_embeddings: Optional[bool] = True,
            ) -> Dict[str, Any]:
    biotrainer_model = BiotrainerModel.from_training_result(training_output_file)
    return biotrainer_model.predict(model_input, save_embeddings, scale_embeddings)


@app.command
def embed(embedder_name: str,
          input_file: Union[str, Path],
          output_dir: Optional[Union[str, Path]] = None,
          reduce: bool = True,
          ) -> str:
    """ Computes embeddings for a given input file into the given output directory. Reduces to per-sequence by default"""
    embedding_api = EmbeddingAPI(embedder_name)
    if output_dir is None:
        output_dir = Path(".")
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    protocol = Protocol.using_per_sequence_embeddings()[0] if reduce else Protocol.using_per_residue_embeddings()[0]
    result_path = embedding_api.compute_embeddings(input_data=input_file, output_dir=output_dir, protocol=protocol)
    return result_path

@app.command
def autoeval(embedder_name: str,
             framework: str,
             min_seq_length: Optional[int] = 0,
             max_seq_length: Optional[int] = 2000,
             use_half_precision: Optional[bool] = False,
             ):
    for progress in autoeval_pipeline(embedder_name=embedder_name,
                                      framework=framework,
                                      min_seq_length=min_seq_length,
                                      max_seq_length=max_seq_length,
                                      use_half_precision=use_half_precision,
                                      ):
        print(progress)


if __name__ == "__main__":
    app()
