# PriorNetworks

## Training scripts to be used with SEML

Configuration which can be edited before submitting the training script to SLURM cluster can be found [here](prior_networks/config_seml.yaml).

The actual training script that will be used the SLURM job can be found [here](prior_networks/trainer_seml.py). This will set up the model with the configuration from the config file, and start the training process.

## Evaluation script

So far the model evaluation resides in a Jupyter Notebook. This can be found [here](prior_networks/EvaluationScript.ipynb).
TODO: Move this to a python file, so that it can be submitted to the SLURM cluster just like training.
