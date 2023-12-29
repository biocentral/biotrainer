from pathlib import Path

import torch
from typing import Union, Optional
from importlib.util import spec_from_file_location, module_from_spec
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5EncoderModel

from .custom_embedder import CustomEmbedder
from .embedding_service import EmbeddingService
from .embedder_interfaces import EmbedderInterface
from .one_hot_encoding_embedder import OneHotEncodingEmbedder
from .huggingface_transformer_embedder import HuggingfaceTransformerEmbedder

__PREDEFINED_EMBEDDERS = {
    "one_hot_encoding": OneHotEncodingEmbedder
}


def get_embedding_service(embeddings_file_path: Union[str, None], embedder_name: Union[str, None],
                          half_precision: Optional[bool] = False,
                          device: Optional[Union[str, torch.device]] = None) -> EmbeddingService:
    if embeddings_file_path is not None:
        # Only for loading
        return EmbeddingService()

    embedder: EmbedderInterface = _get_embedder(embedder_name=embedder_name, half_precision=half_precision,
                                                device=device)
    return EmbeddingService(embedder=embedder)


def _get_embedder(embedder_name: str, half_precision: bool,
                  device: Optional[Union[str, torch.device]]) -> EmbedderInterface:
    # Predefined Embedders
    if embedder_name in __PREDEFINED_EMBEDDERS.keys():
        return __PREDEFINED_EMBEDDERS[embedder_name]()

    # Custom Embedders
    if embedder_name[-3:] == ".py":  # Check that file ends on .py => Python script
        return _load_custom_embedder(embedder_name=embedder_name)

    # Huggingface Transformer Embedders
    torch_dtype = torch.float16 if half_precision else torch.float32
    try:
        tokenizer = AutoTokenizer.from_pretrained(embedder_name, torch_dtype=torch_dtype)
        model = AutoModelForSeq2SeqLM.from_pretrained(embedder_name, torch_dtype=torch_dtype)
    except OSError as os_error:
        raise Exception(f"{embedder_name} could not be found!") from os_error
    except Exception:
        try:
            tokenizer = T5Tokenizer.from_pretrained(embedder_name, torch_dtype=torch_dtype)
            model = T5EncoderModel.from_pretrained(embedder_name, torch_dtype=torch_dtype)
        except Exception as e:
            raise Exception(f"Loading {embedder_name} automatically and as T5Tokenizer failed! Please provide"
                            f"a custom_embedder script for your use-case.") from e
    return HuggingfaceTransformerEmbedder(model=model, tokenizer=tokenizer, half_precision=half_precision,
                                          device=device, name=embedder_name)


def _load_custom_embedder(embedder_name: str) -> EmbedderInterface:
    if not Path(embedder_name).exists():
        raise Exception(f"Custom embedder should be used, but path to script does not exist!\n"
                        f"embedder_name: {embedder_name}")

    if not Path(embedder_name).is_file():
        raise Exception(f"Custom embedder should be used, but path to script is not a file!\n"
                        f"embedder_name: {embedder_name}")

    # Load the module from the file path
    spec = spec_from_file_location("module_name", embedder_name)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    # Check for some problematic modules from external script
    disallow_modules = ["grequests", "requests", "urllib", "os"]
    for name in dir(module):
        if name in disallow_modules:
            raise Exception(f"Module {name} not allowed for custom embedder script!")

    # Find custom embedder in script
    custom_embedder_class = None
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, CustomEmbedder) and not obj.__name__ == "CustomEmbedder":
            custom_embedder_class = obj
            break

    if custom_embedder_class is None:
        raise Exception(f"Did not find custom embedder {embedder_name} in the provided script!")

    if not issubclass(custom_embedder_class, EmbedderInterface):
        raise Exception(f"The provider custom embedder {embedder_name} does not inherit from EmbedderInterface!")

    custom_embedder_instance = custom_embedder_class()
    return custom_embedder_instance


def get_predefined_embedder_names() -> List[str]:
    return list(__PREDEFINED_EMBEDDERS.keys())


__all__ = [
    "EmbeddingService",
    "OneHotEncodingEmbedder",
    "CustomEmbedder",
    "get_embedding_service",
    "get_predefined_embedder_names"
]
