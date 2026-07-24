# LoRA Finetuning on a downstream task example

This example shows how to use [LoRA finetuning](https://doi.org/10.48550/arXiv.2106.09685) on a downstream task. 
Employing (LoRA) finetuning has the potential of improving model performance, while requiring significant more compute 
power, because embeddings have to be re-calculated every epoch and model size increases via the finetuning adapter.
