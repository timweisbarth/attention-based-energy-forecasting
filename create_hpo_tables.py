import numpy as np
import os
import argparse
import pandas as pd


def main():

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
    parser.add_argument('--exp_name', type=str, required=True, default="Exp0", help='Which experiment do you want to create the table of?')
    parser.add_argument('--hpo', type=bool, default=False, help='Is it a HPO experiment of one model only?')
    args = parser.parse_args()

    root_dir = str(os.getcwd()) + f"/results/{args.exp_name}/"
    print(root_dir)
    # Get directories
    dirs = [dirnames for dipath, dirnames, filenames in os.walk(root_dir)][0]

    # Pick directories that are from the relevant experiment
    #dirs = [dir for dir in dirs if args.exp_name in dir] TODO: Check if okay to comment out

    ##################### Create empty MultiIndex DataFrame ################################
    param_name_map = {"learning_rate": "lr", "batch_size": "bs", "e_layers": "el", "d_layers": "dl", "d_model": "dm", "seq_len": "sl", "pred_len": "pl"}
    inv_param_name_map = {v: k for k, v in param_name_map.items()}
    column_metrics_index = ["MSE", "MAE", "Epochs", "Time[min]", "Params[Mio.]", "Max_mem[MB]"]

    # columns for the hpo table in the order that they will be displayed
    column_param_index = ["learning_rate", "batch_size", "e_layers", "d_layers", "d_model", "seq_len", "pred_len"]

    column_index = column_param_index + column_metrics_index
        
    metrics_df = pd.DataFrame(columns=column_index).fillna('-')  # 



    ##############################################################################

    ################ Fill DataFrame with mae and mse #############################

    for dir in dirs:
        for i, subdir in enumerate([dirnames for dipath, dirnames, filenames in os.walk(root_dir + dir + "/")][0]):
            metrics_vals = np.load(root_dir + dir + "/" + subdir + "/_metrics.npy")
            param_values_dict = {}

            # Get the models parameters from the subdir name
            for i, subdir_param in enumerate(subdir.split("_")):
               # Get the mapped param names that are present in current hpo e.g. lr, bs
                mapped_param_names = [param_name_map[column_index] for column_index in column_param_index]

                # Check whether the current subdir param is part of the hpo
                current_param = [mapped_param_name for mapped_param_name in mapped_param_names if mapped_param_name in subdir_param]
                if len(current_param) > 0:
                    _, value = subdir_param.split(f'{current_param[0]}')
                    key, _ = subdir_param.split(f'{value}')
                    param_values_dict[key] = float(value) if key == "lr" else int(float(value))

            params_values_dict = {inv_param_name_map[k]: [v] for k, v in param_values_dict.items()}
            metrics_values_dict = {"MSE": metrics_vals[1], "MAE": metrics_vals[0], 
                              "Epochs": int(metrics_vals[-2]), "Time[min]": int(metrics_vals[-1]/60), 
                              "Params[Mio.]": round(metrics_vals[-4]/(1000**2),2), "Max_mem[MB]": round(metrics_vals[-3], 2)}

            #print(params_values_dict)
            #print(metrics_values_dict)
            values = {**params_values_dict, **metrics_values_dict}
            #print(metrics_df)
            #print(pd.DataFrame.from_dict(values))
            metrics_df = pd.concat([metrics_df, pd.DataFrame.from_dict(values)])
    #print(metrics_df)
    metrics_df = metrics_df.sort_values(by=column_param_index)
    metrics_df = metrics_df.reset_index(drop=True)
    print(metrics_df)
    file_path_csv_metrics = "./results/hpo_table_{}_metrics.csv".format(args.exp_name)
    metrics_df.to_csv(file_path_csv_metrics, header=True, index=True)

    print("\n ------------------ Sorted according to MSE ------------------ \n")
    metrics_df.sort_values(by="MSE", inplace=True)
    print(metrics_df)

    

if __name__ == "__main__":
    main()