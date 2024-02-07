import sys
sys.path.append("./utils/")
from joblib import dump
import time
import torch
import random
import numpy as np
import os
import time

#Modules of src folder
import preproc as pp
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
    df = dl.load_data(args.path, args.from_raw)
    metrics = []

    # Run pipeline for each target and each horizon
    for t in args.targets:
        print(f"------- Starting to train ft{args.forecast_setting} {args.model_name} on {t} for horizons {args.forecast_horizons} ----------", )
        for h in args.forecast_horizons:
            start_time = time.time()
            
            # General preprocessing
            df_train, df_val, df_test = pp.all_preproc_steps(df, t, args.scaler, args.from_raw, args.file_name)

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

                # Train and predict
                model = o.train(X_train, y_train, X_val, y_val, args.model_name)
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
                
                folder_path = "./../checkpoints/" + setting + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                dump(model, folder_path + 'checkpoint.joblib')

            folder_path = "./../results/" + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            if args.save_benchmark:

                np.save(folder_path + '_metrics.npy', np.array([mae, mse, np.nan , train_time]))
                np.save(folder_path + '_pred.npy', preds)
                np.save(folder_path + '_true.npy', truths)
            
            # Plot
            if args.plot:
                # Recover datetime index, undo scaling
                index = df_val.index[args.window_size:]
                preds, truths = eval.inverse_transformations(preds, truths, args.scaler, h)
                #print(index.shape, index[4342])
                v.plot_prediction_vs_truths(preds, truths, args.window_size, h, t, index, args.plot_date, args.days, args.stride, folder_path)

                
                



            
    # Save summary metrics
    eval.print_and_save_benchmark_table(metrics, args)
        
