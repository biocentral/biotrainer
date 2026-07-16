import torch

from biotrainer_core.data_classes import Protocol


def get_dummy_input(protocol: Protocol, embedding_dimension: int):
    """ Get dummy input for a model based on protocol and embedding dimension """
    batch_size = 1
    default_sequence_length = 50
    if protocol in Protocol.using_per_residue_embeddings():
        return torch.rand((batch_size, default_sequence_length, embedding_dimension), dtype=torch.float32)
    return torch.rand((batch_size, embedding_dimension), dtype=torch.float32)
