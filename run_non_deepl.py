import sys
sys.path.append("./utils/")
from joblib import dump
import time
import torch
import random
import numpy as np
import os
import time
import argparse
from sklearn.preprocessing import StandardScaler
from numba import cuda



#Modules of src folder
import preproc as pp
from tools import dotdict
import optimization as o
import data_loader as dl
import eval as eval
import visualizations as v

def pipeline(args):
    """Run the entire pipeline as specified by the args"""
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    df = dl.load_data(args.path)
    metrics = []

    # Run pipeline for each target and each horizon
    for t in args.targets:
        print(f"------- Starting to train {args.model_name} on {t} for horizons {args.forecast_horizons} ----------", )
        for h in args.forecast_horizons:
            
            
            # General preprocessing
            df_train, df_val, df_test = pp.all_preproc_steps(df, t, args.scaler)


            # Check if CUDA is available
            if cuda.is_available():
                # List all available CUDA devices
                for device in cuda.list_devices():
                    print("Found CUDA Device: ", device.name)
                # Set to use the first CUDA device, if available
                device = "cuda:0"
            else:
                print("CUDA is not available. Using CPU instead.")
                device = "cpu"

            if args.model_type == "deepl":
                # Reshape data into inout sequence for nn.lstm module
                (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
                    pp.create_inout_sequence(df_train, df_val, df_test, t[0], h,  args.window_size, args.stride)
        
                train_loader, val_loader, test_loader, val_loader_one = \
                    pp.create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, args.train_params.batch_size)

                # Train and predict
                args.model_params.output_dim = h
                opt = o.Optimization(args, train_loader, val_loader)
                opt.train()
                preds, truths = opt.predict(val_loader)

            else:
                # Reshape data into supervised problem for sklearn module
                (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
                pp.make_supervised(df_train, df_val, df_test, t, h, args.window_size, args.stride, args.cols_to_lag)

                start_time = time.time()

                # Train and predict
                model = o.train(X_train, y_train, X_val, y_val, args.model_name, device)
                preds = model.predict(X_val)
                truths = y_val
                #print(preds.shape)
                #print(y_val.shape)
            
            train_time = time.time() - start_time
            # Evaluate
            mae, mse = eval.calc_metrics(preds, truths)
            metrics.append({"target": t[0], "horizon": h, "mae":mae, "mse": mse})

            setting = "ft{}_{}_{}_sl{}_ll{}_pl{}_{}_eb{}_{}".format(
                "S" if len(t) == 1 else "M",
                "smard",
                t[0] if len(t) == 1 else "",
                args.window_size,
                0,
                h,
                args.model_name,
                "timeF",
                args.experiment_name,
            )

            

            # Save model
            if args.save_model:
                
                folder_path = "/Users/timw/Documents/University/thesis_2/checkpoints/" + setting + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                dump(model, folder_path + 'checkpoint.joblib')
            #print(os.getcwd())
            folder_path = "/Users/timw/Documents/University/thesis_2/results/" + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            if args.save_benchmark:

                np.save(folder_path + '_metrics.npy', np.array([mae, mse, np.nan , train_time]))
                #np.save(folder_path + '_pred.npy', preds)
                #np.save(folder_path + '_true.npy', truths)
            
            # Plot
            if args.plot:
                # Recover datetime index, undo scaling
                index = df_val.index[args.window_size:]
                preds, truths = eval.inverse_transformations(preds, truths, args.scaler, h)
                v.plot_prediction_vs_truths(preds, truths, args.window_size, h, t, index, args.plot_date, args.days, args.stride, folder_path)

                
                



            
    # Save summary metrics
    eval.print_and_save_benchmark_table(metrics, args)

if __name__ == "__main__":

    # Executing this from the shell will run pipeline for xgb with below setting:
    args = dotdict({})
    args.model_params = dotdict({})
    args.train_params = dotdict({})

    args.experiment_name = "Exp0"

    # Data loading
    args.from_raw = True
    args.file_name = "smard_data.csv"

    # Preprocessing
    args.scaler_name = "std"

    # Model and its hyperparameters
    args.model_name = "xgb"
    args.model_params = None
    args.train_params = None

    # Prediction
    args.cols_to_lag = ['load', 'solar_gen', 'wind_gen']
    args.targets = [['load'], ['solar_gen'], ['wind_gen'], ['load', 'solar_gen', 'wind_gen']]
    args.window_size = 12
    args.stride = 1 # Has to be <= min(window_size, forecast_horizon) and stride * integer = window_size,
    # and stride * integer2 = forecast_horizon
    args.lead_time = 0 # TODO: Not working yet
    args.forecast_horizons = [24] #  48, 96, 192, 336, 720]

    # Plotting
    args.plot = True
    args.plot_date = '2021-07-01'
    args.days = 10

    # Save model
    args.save_model = False
    args.save_benchmark = True

    # Composition of arguments given
    from_raw_folder = "raw" if args.from_raw else "preproc_non_deepl"
    args.path = f"./data/{from_raw_folder}/{args.file_name}"
    args.scaler = {"std":StandardScaler()}[args.scaler_name]

    pipeline(args)
        
