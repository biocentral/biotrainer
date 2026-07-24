# biotrainer-core

A lightweight Python package providing shared data classes, I/O utilities, and helper functions for protein-related
projects.

## Purpose

`biotrainer-core` is designed as a minimal, dependency-light foundation that can be reused across multiple protein ML
projects (e.g. [biotrainer](https://github.com/sacdallago/biotrainer)) without pulling in heavy frameworks like PyTorch
or Transformers.

## Features

### Data Classes

Pydantic-based data models shared across protein ML workflows:

- **Sequence data** — representations for protein sequences and their metadata
- **Protocols** — supported task types (residue-level, sequence-level, contact prediction, …)
- **Metrics** — structured containers for evaluation metrics
- **Model results & predictions** — standardised output formats for biotrainer models
- **Embedding statistics** — summary statistics over embedding spaces
- **AutoEval** — data classes for automated evaluation tasks, reports, and benchmark datasets (FLIP, PBC)

### H5 File Handling

Utilities for reading and writing HDF5 files used to store protein embeddings:

- `H5Database` — high-level interface for embedding databases
- Low-level helpers for chunked I/O and dataset management

### Functions

Pure utility functions with no side effects:

- **Hashing** — deterministic sequence hashing
- **Seeding** — reproducible random seed management
- **Ranking** — ranking helpers for evaluation
- **Bootstrapping** — confidence interval estimation via bootstrapping

### Input Files

- **FASTA parsing** — read and validate FASTA files into typed sequence objects

## Installation

```bash
pip install biotrainer-core
```

## License

*MIT* — see [LICENSE](LICENSE) for details.
