import numpy as np
import os
import argparse
import pandas as pd

def main():

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
    parser.add_argument('--exp_name', type=str, required=True, default="Exp0", help='Which experiment do you want to create the table of?')
    args = parser.parse_args()

    root_dir = str(os.getcwd()) + f"/results/{args.exp_name}/"

    # Get directories
    dirs = [dirnames for dipath, dirnames, filenames in os.walk(root_dir)][0]

    # Pick directories that are from the relevant experiment
    dirs = [dir for dir in dirs if args.exp_name in dir]

    ##################### Create empty MultiIndex DataFrame ################################
    # Define the levels for the MultiIndex of the rows
    targets = ["multi", "load", "solar", "wind"]  
    horizons = ['24', '96', '192', '336', '720'] 

    # Define the levels for the MultiIndex of the columns
    models = ['iTransformer', 'PatchTST', 'Autoformer', 'Informer', 'Transformer', 'TSMixer', "DLinear", "LSTM", "XGBoost", "Linear Regression", "Dummy"]  
    metrics = ['MSE', 'MAE'] 

    # Create the MultiIndex for the rows
    row_index = pd.MultiIndex.from_product([targets, horizons], names=['Target', 'Horizon'])

    # Create the MultiIndex for the columns
    column_index = pd.MultiIndex.from_product([models, metrics], names=['Model', 'Metric'])

    metrics_df = pd.DataFrame(index=row_index, columns=column_index).fillna('-')  # 

    epoch_time_df = metrics_df.copy()
    epoch_time_df = epoch_time_df.rename(columns={'MAE': 'time[min]', 'MSE': 'epochs'}, level='Metric')

    modelsize_maxmemory_df = metrics_df.copy()
    modelsize_maxmemory_df = modelsize_maxmemory_df.rename(columns={'MAE': 'max_mem[MB]', 'MSE': 'params[Mio.]'}, level='Metric')

    std_df = metrics_df.copy()
    std_df = std_df.rename(columns={'MAE': 'std_MAE', 'MSE': 'std_MSE'}, level='Metric')

    ##############################################################################

    ################ Fill DataFrame with mae and mse #############################

    targets_file_name = ['ftM', 'load', 'solar', 'wind']
    horizons_file_name = ['pl24','pl96','pl192','pl336','pl720']
    models_file_name = ['iTransformer', 'PatchTST', 'Autoformer', 'Informer', 'Transformer', 'TSMixer', 'DLinear', "LSTM", "xgb", "linreg", "dummy"]

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
                 break
        
        maes = []
        mses = []
        number_of_epochs_for_trainings = []
        total_train_times = []
        modelsizes = []
        max_memorys = []

        for i, subdir in enumerate([dirnames for dipath, dirnames, filenames in os.walk(root_dir + dir + "/")][0]):
            metrics_vals = np.load(root_dir + dir + "/" + subdir + "/_metrics.npy")
            maes.append(metrics_vals[0])
            mses.append(metrics_vals[1])
            number_of_epochs_for_trainings.append(metrics_vals[-2])
            total_train_times.append(metrics_vals[-1])
            max_memorys.append(metrics_vals[-3])
            modelsizes.append(metrics_vals[-4])

        
        # metrics_df
        metrics_df.loc[(target, horizon), (model, 'MAE')] = round(sum(maes)/len(maes), 3)
        metrics_df.loc[(target, horizon), (model, 'MSE')] = round(sum(mses)/len(mses), 3)

        # epoch_train_time_df
        avg_epochs_for_training = sum(number_of_epochs_for_trainings)/len(number_of_epochs_for_trainings)
        avg_total_train_time = sum(total_train_times)/len(total_train_times)
        epoch_time_df.loc[(target, horizon), (model, 'epochs')] = int(avg_epochs_for_training) if not np.isnan(avg_epochs_for_training) else '-'
        epoch_time_df.loc[(target, horizon), (model, 'time[min]')] = round(avg_total_train_time / 60)

        # modelsize_maxmemory_df
        avg_modelsize = sum(modelsizes)/len(modelsizes)
        avg_max_memory = sum(max_memorys)/len(max_memorys)
        modelsize_maxmemory_df.loc[(target, horizon), (model, 'params[Mio.]')] = round(avg_modelsize/10**6,1) if not np.isnan(avg_modelsize) else '-'
        modelsize_maxmemory_df.loc[(target, horizon), (model, 'max_mem[MB]')] = int(avg_max_memory) if not np.isnan(avg_max_memory) else '-'

        # std_df
        std_df.loc[(target, horizon), (model, 'std_MAE')] = round(np.std(maes), 3) if len(maes) > 1 else '-'
        std_df.loc[(target, horizon), (model, 'std_MSE')] = round(np.std(mses), 3) if len(mses) > 1 else '-'



    #print("epochs:", epoch_time_df.loc[("multi", "96"), ("XGBoost", 'epochs')])
    #print(metrics_df)
    #print(epoch_time_df)
    #print(modelsize_maxmemory_df)
    #print(std_df)

    # Save df as .csv and .tex
    latex_table = metrics_df.to_latex()
    

    def apply_formatting(val, lowest, second_lowest):
        if val == lowest:
            return r'\textbf{' + str(val) + '}'
        elif val == second_lowest:
            return r'\underline{' + str(val) + '}'
        else:
            return str(val)
    
    for row in metrics_df.index:
        for metric in metrics:
            #print(metric)
            values = metrics_df.loc[row, (slice(None), metric)]
            sorted_values = values.sort_values()
            if len(sorted_values) > 1:
                lowest, second_lowest = sorted_values[:2]
                formatted_values = values.apply(lambda x: apply_formatting(x, lowest, second_lowest))
                metrics_df.loc[row, (slice(None), metric)] = formatted_values
    # Convert the modified DataFrame to LaTeX
                
    stacked_df = metrics_df.stack(level=0)

    # Step 3: Unstack the previously stacked level
    final_df = stacked_df.unstack(level=0)

    final_df = final_df.swaplevel(axis=1).sort_index(axis=1)
    final_df = final_df.swaplevel(axis=0).sort_index(axis=0)
    #final_df = metrics_df.transpose()
    latex_table_formatted = final_df.to_latex(escape=False)



    file_path_csv_metrics = "./results/benchmark_table_{}_metrics.csv".format(args.exp_name)
    file_path_tex_metrics = "./results/benchmark_table_{}_metrics.tex".format(args.exp_name)
    file_path_tex_metrics_formatted = "./results/benchmark_table_{}_metrics_formatted.tex".format(args.exp_name)

    file_path_csv_epoch_time = "./results/benchmark_table_{}_epoch_time.csv".format(args.exp_name)

    file_path_csv_modelsize_maxmemory = "./results/benchmark_table_{}_modelsize_maxmemory.csv".format(args.exp_name)

    file_path_csv_std = "./results/benchmark_table_{}_std.csv".format(args.exp_name)

    metrics_df.to_csv(file_path_csv_metrics, header=True, index=True)
    epoch_time_df.to_csv(file_path_csv_epoch_time, header=True, index=True)
    modelsize_maxmemory_df.to_csv(file_path_csv_modelsize_maxmemory, header=True, index=True)
    std_df.to_csv(file_path_csv_std, header=True, index=True)

    with open(file_path_tex_metrics, 'w') as file:
        file.write(latex_table)
    with open(file_path_tex_metrics_formatted, 'w') as file:
        file.write(latex_table_formatted)

if __name__ == "__main__":
    main()