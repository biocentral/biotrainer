from .random_embedder import RandomEmbedder
from .blosum62_embedder import Blosum62Embedder
from .aa_ontology_embedder import AAOntologyEmbedder
from .one_hot_encoding_embedder import OneHotEncodingEmbedder

BASELINE_EMBEDDERS = {
    "one_hot_encoding": OneHotEncodingEmbedder,
    "random_embedder": RandomEmbedder,
    "AAOntology": AAOntologyEmbedder,
    "blosum62": Blosum62Embedder,
}

__all__ = ["BASELINE_EMBEDDERS", "AAOntologyEmbedder", "OneHotEncodingEmbedder", "RandomEmbedder", "Blosum62Embedder"]
