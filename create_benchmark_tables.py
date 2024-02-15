import numpy as np
import os
import argparse
import pandas as pd

def main():

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
    parser.add_argument('--exp_name', type=str, required=True, default="Exp", help='Which experiment do you want to create the table of?')
    args = parser.parse_args()

    root_dir = str(os.getcwd()) + "/results/"

    # Get directories
    dirs = [dirnames for dipath, dirnames, filenames in os.walk(root_dir)][0]

    # Pick directories that are from the relevant experiment
    dirs = [dir for dir in dirs if args.exp_name in dir]

    ##################### Create empty MultiIndex DataFrame ################################
    # Define the levels for the MultiIndex of the rows
    targets = ["multi", "load", "solar", "wind"]  
    horizons = ['24', '48', '96', '192', '336', '720'] 

    # Define the levels for the MultiIndex of the columns
    models = ['Autoformer', 'Informer', 'Transformer', "LSTM", "XGBoost", "Linear Regression", "Dummy"]  
    metrics = ['MSE', 'MAE'] 

    # Create the MultiIndex for the rows
    row_index = pd.MultiIndex.from_product([targets, horizons], names=['Target', 'Horizon'])

    # Create the MultiIndex for the columns
    column_index = pd.MultiIndex.from_product([models, metrics], names=['Model', 'Metric'])

    metrics_df = pd.DataFrame(index=row_index, columns=column_index).fillna('-')  # 

    epoch_time_df = metrics_df.copy()
    epoch_time_df = epoch_time_df.rename(columns={'MAE': 'time[min]', 'MSE': 'epochs'}, level='Metric')

    ##############################################################################

    ################ Fill DataFrame with mae and mse #############################

    targets_file_name = ['ftM', 'load', 'solar', 'wind']
    horizons_file_name = ['pl24', 'pl48','pl96','pl192','pl336','pl720']
    models_file_name = ['Autoformer', 'Informer', 'Transformer', "LSTM", "xgb", "linreg", "dummy"]
    #metrics_df.loc[('multi', '96'), ('Informer', 'MAE')] = 2.3124
    #print(metrics_df)
    for dir in dirs:

        for t, t_file_name in zip(targets, targets_file_name):
            if t_file_name in dir:
                 target = t
        for h, h_file_name in zip(horizons, horizons_file_name):
            if h_file_name in dir:
                 horizon = h
        for m, m_file_name in zip(models, models_file_name):
            if m_file_name in dir:
                 model = m

        metrics = np.load(root_dir + dir + "/_metrics.npy")

        # MAE, MSE
        mae = metrics[0]
        mse = metrics[1]
        metrics_df.loc[(target, horizon), (model, 'MAE')] = round(mae, 3)
        metrics_df.loc[(target, horizon), (model, 'MSE')] = round(mse, 3)

        # Early stop
        number_of_epochs_for_training = metrics[-2]
        total_train_time = metrics[-1]
        epoch_time_df.loc[(target, horizon), (model, 'epochs')] = number_of_epochs_for_training
        epoch_time_df.loc[(target, horizon), (model, 'time[min]')] = round(total_train_time / 60)
        
    print(metrics_df)
    print(epoch_time_df)

    # Save df as .csv and .tex
    latex_table = metrics_df.to_latex()
    file_path_csv_metrics = "./results/benchmark_table_{}_metrics.csv".format(args.exp_name)
    file_path_tex_metrics = "./results/benchmark_table_{}_metrics.tex".format(args.exp_name)
    file_path_csv_epoch_time = "./results/benchmark_table_{}_epoch_time.csv".format(args.exp_name)

    metrics_df.to_csv(file_path_csv_metrics, header=True, index=True)
    epoch_time_df.to_csv(file_path_csv_epoch_time, header=True, index=True)

    with open(file_path_tex_metrics, 'w') as file:
        file.write(latex_table)


if __name__ == "__main__":
    main()