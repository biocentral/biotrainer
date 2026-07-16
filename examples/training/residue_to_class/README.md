# residue_to_class example

This example shows how to use the residue_to_class protocol.
An `input.fasta` file with the amino acid sequences and the per-residue labels (`TARGET=`) has to be provided.
For more information, see [data standardization](../../../docs/data_standardization.md#residue_to_class). 

Execute the example (from the base directory):
```bash
biotrainer train --config examples/training/residue_to_class/config.yml
```
