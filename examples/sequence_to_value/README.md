# sequence_to_value example

The sequence_to_value protocol implements a regression task on the sequence level. This means to predict
a numerical value for every sequence provided. Note that this protocol must use 
`loss_choice: mean_squared_error` as a loss function currently.

Execute the example (from the base directory):
```bash
biotrainer train --config examples/sequence_to_value/config.yml
```
