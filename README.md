# Thesis

This is the repository of my thesis "Attention-Based Multi-Horizon Energy Forecasting: A Case Study on the Transmission Grid" and is based on the Github repository of Autoformer. 

## Overview

The ./scripts/Expxx.../ folder contains shell scripts and IPython notebooks with the configurations of each model run. It also contains an IPython notebook with the analysis of the experiment. The mapping between the chapters, folders, figures and tables can be found below: 

| Chapter  | Repository folder | Figure | Tables |
| ----------------------------- | ------------------------------------------------- | ----------------- | --------- |
| Chapter 2.4: Data             | Exp0_.../data_exploration_smard_DE.ipynb          | Figure 2.6        |           |
| Chapter 3.1: Benchmark        | Exp1_.../analysis_default_benchmark_ED.ipynb      | Figure 3.1        | Table 3.1 |
|                               | Exp1.2_...                                        |                   |           |
|                               | Exp2_...                                          |                   |           |
| Chapter 3.2: HPO for Benchmark| Exp0_.../data_exploration_smard_DE.ipynb          | Figure 3.2        |           |
|                               | Exp2.xx_.../....sh scripts                        |                   | Table 3.2 |
|                               | Exp2.5_.../analysis_HPO_transformer_h24_ED.ipynb  | Figure 3.3, 3.4   | Table 3.3 |
|                               | Exp2.6_.../analysis_HPO_transformer_h336_ED.ipynb | Figure 3.3, 3.4   | Table 3.3 |
|                               | Exp2.8_.../analysis_HPO_itransformer_h24_ED.ipynb | Figure 3.5, 3.6   | Table 3.4 |
|                               | Exp2.9_.../analysis_HPO_itransformer_h336_ED.ipynb| Figure 3.5, 3.6   | Table 3.4 |
|                               | Exp2.11_.../analysis_HPO_patchtst_h192_ED.ipynb   | Figure 3.7        | Table 3.5 |
|                               | Exp2.10_.../analysis_HPO_baselines_ED.ipynb       | Table 3.6         |           |
| Chapter 3.3: Benchmark EED    | Exp3.1_.../analysis_experiments_weather_EED.ipynb | Figure 3.9        | Table 3.7 |
| Chapter 3.4: HPO EED          | Exp3.2_.../analysis_HPO_transformer_EED.ipynb     | Figure 3.10       |           |
|                               | Exp3.3_.../analysis_HPO_transformer_2_EED.ipynb   | Figure 3.10       |           |
|                               | Exp3.4_.../analysis_HPO_itransformer_EED.ipynb    |                   | Table 3.8 |
|                               | Exp3.5_.../analysis_HPO_LSTM_EED.ipynb            |                   | Table 3.9 |
| Chapter 4: Test set           | Exp4_.../analysis_benchmark_test_set.ipynb        | Figure 3.12       | Table 3.10|

# How to use

Each shell scripts can be executed on the cluster via the sbatch command in order to reproduce an experiment. The IPython notebooks can be executet on a local machine. New experiment can be conducted following the convention of existing shell scripts. For this create a new folder in ./scritps/ following the naming convention Expxx_xx_xx and execute it from the thesis2 folder. Experiments on new data require a new class in the data_loader.py file.

The results of the experiment can be loaded in an csv file. If the experiment contains different models, run python3 create_benchmark_tables.py --exp_name "Expxx". If multiple runs for one model are conducted, run python3 create_hpo_tables.py --exp_name "Expxx". These created csv files can be analysed as shown in the ./scripts/Expxx folders.

# Troubleshooting

If the creation of the csv files fails, it most likely due to certain model names are expected but not present in the experiment. Thus the create_benchmark or create_hpo files need slight modifications. 

The analysis IPython files may raise errors due to certain features present/absent in the csv file. This can be fixed by dropping/adding them manually.

# Details

This project contains two pipelines. The pytorch architectures (Attention-based models, DLinear, LSTM and TSMixer) are exectuted via shell scripts that call run.py which in turn calls exp_main.py. The second pipeline is for XGBoost and Linear Regression and is executed through shell scripts or IPython notebooks that call run_non_deepl.py. The checkpoints folder contains the checkpoints of the model parameters (if saved). However in most cases it only contains the train vs. val loss plot of each model run. The results folder contains the forecast and ground truth of each experiment. The naming convention of the folders can be found in the setting variable of run.py which is based on the parser arguments of the same document. 

# Map chapters to experiments, figures and tables






