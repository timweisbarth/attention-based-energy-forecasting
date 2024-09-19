# "Attention-Based Multi-Horizon Energy Forecasting: A Case Study on the German Electricity Grid"

Repository of my thesis "Attention-Based Multi-Horizon Energy Forecasting: A Case Study on the German Transmission Grid", supervised by Dr. Nicole Ludwig from the [Machine Learning in Sustainable Energy Systems](https://www.mlsustainableenergy.com) group at the University of Tuebingen.

# Abstract

Climate change has triggered a transformation of the German energy grid towards more volatile renewable energies. This requires accurate forecasts of energy  generation and consumption over different time horizons, as the grid still needs to be balanced at all times. It is currently unclear which models, and thereby inductive biases, are best suited for the specific characteristics of energy time series forecasting. This is especially true after a simple linear layer model, called DLinear, outperformed all sophisticated attention-based models in 2022 on common benchmark datasets. Thus, this work compares different generations of attention-based models (PatchTST, iTransformer, Autoformer, Informer and Transformer) against a wide range of baselines (DLinear, Linear Regression, TSMixer, LSTM and XGBoost). This is done in a univariate task and a multivariate task that requires good modeling of covariate effects. The results show that only attention-based models provide the best evaluation metric on the test set. They outperform the best baseline, which is XGBoost, by $25\%$ in the univariate and 15% in the multivariate task. Surprisingly, the (vanilla) Transformer is the best among the attention-based models. Along the way, the benchmarks and optimizations provide valuable insights about model size as well as current research on time series transformers and their limitations.

# How to use

The ./scripts/Expxx.../ folders are the central objects of this repository. They contain shell scripts and IPython notebooks with the configurations of each model run. They also contains an IPython notebook with the analysis. The mapping between the thesis' chapters, figures and tables to the repositories folders can be found in the file mapping.md.

Each Expxx in ./scripts has a corresponding folder in ./checkpoints and ./results. Checkpoints contains the train vs. val plot of each model run and (if saved) the model weights. The results folder contains the forecast and ground truth of each experiment.

## Details

This project contains two pipelines. The pytorch architectures (Attention-based models, DLinear, LSTM and TSMixer) are exectuted via shell scripts that call run.py which in turn calls exp_main.py. The second pipeline is for XGBoost and Linear Regression and is executed through shell scripts or IPython notebooks that call run_non_deepl.py.

The general framework and implementations of Trans- and Autoformer are taken from the official [Autoformer](https://github.com/thuml/Autoformer.git) repository. PatchTST and DLinear are taken from the official [PatchTST](https://github.com/yuqinie98/PatchTST.git) repository. The other models are taken from their respective repositories: [Informer](https://github.com/zhouhaoyi/Informer2020.git), [iTransformer](https://github.com/thuml/iTransformer.git), and [TSMixer](https://github.com/ditschuk/pytorch-tsmixer.git). The remaining models are implemented by sklearn, xgboost, or by hand (for example the LSTM).







