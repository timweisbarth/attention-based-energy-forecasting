# "Attention-Based Multi-Horizon Energy Forecasting: A Case Study on the German Transmission Grid"

This is the repository of my thesis "Attention-Based Multi-Horizon Energy Forecasting: A Case Study on the German Transmission Grid" and is based on the Github repository of Autoformer. 

## Overview

The ./scripts/Expxx.../ folder contains shell scripts and IPython notebooks with the configurations of each model run. It also contains an IPython notebook with the analysis. The mapping between the thesis' chapters, figures and tables to the repositories folders can be found in the file mapping.md.

The checkpoints folder contains the checkpoints of the model parameters (if saved). However in most cases it only contains the train vs. val loss plot of each model run. The results folder contains the forecast and ground truth of each experiment. The naming convention of the folders can be found in the setting variable of run.py which is based on the parser arguments of the run.py file. 

# How to use

In order to reproduce an experiment, executed the ./Expxx_.../....sh file via sbatch on the cluster. The IPython notebooks can be executed on a local machine. New experiment can be conducted following the convention of existing shell scripts. For this create a new folder in ./scritps/ following the naming convention Expxx_... . Experiments on new data require a new class in the data_loader.py file.

The results of the experiment can be loaded in an csv file. If the experiment contains different models, run python3 create_benchmark_tables.py --exp_name "Expxx". If multiple runs for one model are conducted, run python3 create_hpo_tables.py --exp_name "Expxx". These created csv files can be analysed as shown in the ./scripts/Expxx folders.

# Troubleshooting

If the creation of the csv files fails, it most likely due to certain model names are expected but not present in the experiment. Thus the create_benchmark or create_hpo files need slight modifications. 

The analysis IPython files may raise errors due to certain features present/absent in the csv file. This can be fixed by dropping/adding them manually.

# Details

This project contains two pipelines. The pytorch architectures (Attention-based models, DLinear, LSTM and TSMixer) are exectuted via shell scripts that call run.py which in turn calls exp_main.py. The second pipeline is for XGBoost and Linear Regression and is executed through shell scripts or IPython notebooks that call run_non_deepl.py.






