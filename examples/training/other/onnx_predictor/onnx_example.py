from biotrainer.embedding import OneHotEncodingEmbedder
from biotrainer.training.inference import Inferencer
from biotrainer_core.data_classes import Protocol

out_file_path_res = '../../residue_to_class/output/out.yml'


def onnx_save():
    inferencer, out_file = Inferencer.create_from_out_file(out_file_path=out_file_path_res)
    converted_checkpoint_paths = inferencer.convert_to_onnx()
    return converted_checkpoint_paths


def onnx_load(model_path: str):
    # Embed
    sequences = [
        "PROVTEIN",
        "SEQVENCESEQVENCE"
    ]

    embedder = OneHotEncodingEmbedder()
    embeddings = list(embedder.embed_many(sequences))

    # Predict ONNX
    onnx_result = Inferencer.from_onnx_with_embeddings(model_path=model_path, embeddings=embeddings,
                                                       protocol=Protocol.sequence_to_class)
    print(onnx_result)

    # Double check against inferencer predictions
    inferencer, out_file = Inferencer.create_from_out_file(out_file_path=out_file_path_res)
    print(inferencer.from_embeddings(embeddings=embeddings))


def main(args=None):
    converted_checkpoint_paths = onnx_save()
    onnx_load(converted_checkpoint_paths[0])


if __name__ == '__main__':
    main()
