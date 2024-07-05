import sys
sys.path.append("./utils/")
from joblib import dump
import time
import torch
import random
import numpy as np
import os
import time
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

#Modules of src folder
import preproc as pp
from tools import dotdict
import optimization as o
import data_loader as dl
import eval as eval
import visualizations as v

def pipeline(args):
    """
    Run the entire pipeline as specified by the args.
    Called from .ipynb files in ./scripts/Expxx_.../ 
    for Linear Regression, Ridge, Naive, Repeat.
    Called as main() from .sh scripts in 
    ./scripts/Expxx_.../ for XGBoost. (see below)

    """
    # Fix seed for reproducibility
    fix_seed = args.fix_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    df = dl.load_data(args.path)
    metrics = []

    # Run ii times in case of HPO
    for ii in range(args.itr):
        # Run pipeline for each target and each horizon
        for t in args.targets:
            print(f"------- Starting to train {args.model_name} on {t} for horizons {args.forecast_horizons} ----------", )
            for h in args.forecast_horizons:

                # General preprocessing
                df_train, df_val, df_test = pp.all_preproc_steps(df, t, args.scaler, args.window_size, args.final_run_train_on_train_and_val)

                # Get device
                if torch.cuda.is_available():
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(0) 
                    device = torch.device('cuda:{}'.format(0))
                    print('Use GPU: cuda:{}'.format(0))
                else:
                    device = "cpu"
                
                # Reshape data into supervised problem for sklearn module
                (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
                pp.make_supervised(df_train, df_val, df_test, t, h, args.window_size, args.stride, args.cols_to_lag, args.point_forecast)
                print("-----------------")
                print("X_train", X_train.shape)
                print("y_train", y_train.shape)
                start_time = time.time()

                # Train and predict
                model = o.train(X_train, y_train, X_val, y_val, args.model_name, device, args.train_params)
                if args.model_name == "xgb":
                    dval = xgb.DMatrix(X_val)
                    preds = model.predict(dval)
                else:
                    preds = model.predict(X_val)
                truths = y_val

                train_time = time.time() - start_time

                # Evaluate
                mae, mse = eval.calc_metrics(preds, truths)

                metrics.append({"target": t[0], "horizon": h, "mae":mae, "mse": mse}) #legacy, used for print at bottom

                # Setting for saving and plotting
                setting = "ft{}_{}_{}_sl{}_ll{}_pl{}_{}_hpo{}_eb{}_{}_iter{}".format(
                    "S" if len(t) == 1 else "M",
                    "smard",
                    t[0] if len(t) == 1 else "",
                    args.window_size,
                    0,
                    h,
                    args.model_name,
                    args.train_params.run,
                    "timeF",
                    args.experiment_name,
                    ii
                )

                # Save model
                if args.save_model:

                    folder_path = "./checkpoints/" + args.experiment_name + '/' + setting + '/'
                    if not os.path.exists(folder_path):

                        os.makedirs(folder_path)
                    dump(model, folder_path + 'checkpoint.joblib')

                # Create folder if plot or save_benchmark is True
                if args.save_benchmark or args.plot:
                    folder_path = "./results/" + args.experiment_name + '/' + setting.split("_iter", 1)[0] + '/' + setting + '/'
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)

                # Save benchmark
                if args.save_benchmark:
                    if torch.cuda.is_available():
                        max_memory = torch.cuda.max_memory_allocated()
                    else:
                        max_memory = np.nan
                    model_size =np.nan
                    epochs= np.nan
                    np.save(folder_path + '_metrics.npy', np.array([mae, mse, model_size, max_memory, epochs, train_time]))

                # Plot
                if args.plot:
                    # Recover datetime index, undo scaling
                    index = df_val.index[args.window_size:]
                    preds, truths = eval.inverse_transformations(preds, truths, args.scaler, h)
                    v.plot_prediction_vs_truths(preds, truths, args.window_size, h, t, index, args.plot_date, args.days, args.stride, folder_path)
                
                # Reset torch memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_max_memory_allocated()

            # Save summary metrics
            eval.print_and_save_benchmark_table(metrics)

if __name__ == "__main__":

    # Executing this from the shell will run pipeline for xgb with below setting:
    args = dotdict({})
    args.model_params = dotdict({})
    args.train_params = dotdict({})

    args.fix_seed = 2024
    args.itr = 1

    # 1 run means run in default setting, multiple runs means test different HPOs as defined in otimization.py
    number_of_runs = 1

    # Preprocessing
    args.scaler_name = "std"

    # Model and its hyperparameters
    args.model_name = "xgb"
    args.model_params = None
    #args.train_params = None

    # Prediction
    args.point_forecast = True
    args.forecast_setting = "both"

    ########################################################################################
    ####### Below: Uncomment sectionwise to reproduce the results of the thesis ############
    ########################################################################################

    ###################### Exp 1 ##########################
    args.experiment_name, args.final_run_train_on_train_and_val = "Exp1", False

    args.file_name = "smard_data_DE.csv"
    args.cols_to_lag = [
        'load', 'solar_gen', 'wind_gen',
    ]
    args.targets = [[
        'load',
        'solar_gen',
        'wind_gen',
        ['load', 'solar_gen', 'wind_gen'],
    ]]

    args.window_size = 336

    ################### Exp3.1 and 4 (MultiXL) ##########################

    #args.experiment_name, args.final_run_train_on_train_and_val = "Exp3.1", False
    #args.experiment_name, args.final_run_train_on_train_and_val = "Exp4", True
    #args.file_name = "smard_plus_weather_without_LUandAT.csv"
    #args.cols_to_lag = [
    #    'load_DE', 'solar_gen_DE', 'wind_gen_DE',
    #    'lat54.125_lon7.375_u100', 'lat54.125_lon7.375_v100', 'lat54.125_lon7.375_t2m', 'lat54.125_lon7.375_ssrd',
    #    'lat54.125_lon10.375_u100', 'lat54.125_lon10.375_v100', 'lat54.125_lon10.375_t2m', 'lat54.125_lon10.375_ssrd',
    #    'lat54.125_lon13.375_u100', 'lat54.125_lon13.375_v100', 'lat54.125_lon13.375_t2m', 'lat54.125_lon13.375_ssrd',
    #    'lat52.125_lon7.375_u100', 'lat52.125_lon7.375_v100', 'lat52.125_lon7.375_t2m', 'lat52.125_lon7.375_ssrd',
    #    'lat52.125_lon10.375_u100', 'lat52.125_lon10.375_v100', 'lat52.125_lon10.375_t2m', 'lat52.125_lon10.375_ssrd',
    #    'lat52.125_lon13.375_u100', 'lat52.125_lon13.375_v100', 'lat52.125_lon13.375_t2m', 'lat52.125_lon13.375_ssrd',
    #    'lat50.125_lon7.375_u100', 'lat50.125_lon7.375_v100', 'lat50.125_lon7.375_t2m', 'lat50.125_lon7.375_ssrd',
    #    'lat50.125_lon10.375_u100', 'lat50.125_lon10.375_v100', 'lat50.125_lon10.375_t2m', 'lat50.125_lon10.375_ssrd',
    #    'lat50.125_lon13.375_u100', 'lat50.125_lon13.375_v100', 'lat50.125_lon13.375_t2m', 'lat50.125_lon13.375_ssrd',
    #    'lat48.125_lon7.375_u100', 'lat48.125_lon7.375_v100', 'lat48.125_lon7.375_t2m', 'lat48.125_lon7.375_ssrd', 
    #    'lat48.125_lon10.375_u100', 'lat48.125_lon10.375_v100', 'lat48.125_lon10.375_t2m', 'lat48.125_lon10.375_ssrd',
    #    'lat48.125_lon13.375_u100', 'lat48.125_lon13.375_v100', 'lat48.125_lon13.375_t2m', 'lat48.125_lon13.375_ssrd',
    #    'load_DE_50Hertz', 'load_DE_Amprion', 'load_DE_TenneT', 'load_DE_TransnetBW', 
    #    'solar_gen_DE_50Hertz',  'solar_gen_DE_Amprion', 'solar_gen_DE_TenneT', 'solar_gen_DE_TransnetBW',
    #    'wind_gen_DE_50Hertz', 'wind_gen_DE_Amprion', 'wind_gen_DE_TenneT', 'wind_gen_DE_TransnetBW'
    #]
    #args.targets = [[
    #    'load_DE', 'solar_gen_DE', 'wind_gen_DE',
    #    'load_DE_50Hertz', 'load_DE_Amprion', 'load_DE_TenneT', 'load_DE_TransnetBW',
    #    'solar_gen_DE_50Hertz', 'solar_gen_DE_Amprion', 'solar_gen_DE_TenneT',  'solar_gen_DE_TransnetBW', 
    #    'wind_gen_DE_50Hertz', 'wind_gen_DE_Amprion',  'wind_gen_DE_TenneT', 'wind_gen_DE_TransnetBW'
    #]]
    #args.window_size = 96

    ########################### Exp4 (Load) ##########################
    #args.experiment_name, args.final_run_train_on_train_and_val = "Exp4", True
    #args.file_name = "smard_data_DE.csv"
    #args.cols_to_lag = [
    #    'load', 'solar_gen', 'wind_gen',
    #]
    #args.targets = [[
    #    'load'
    #]]

    #args.window_size = 336


    #################################################################
    ###################### Always the same ##########################
    #################################################################

    args.stride = 1 # Has to be <= min(window_size, forecast_horizon) and stride * integer = window_size,
    # and stride * integer2 = forecast_horizon
    args.lead_time = 0 # TODO: Not working yet
    args.forecast_horizons = [24, 96, 192, 336, 720]
    

    # Plotting
    args.plot = False
    args.plot_date = '2021-07-01'
    args.days = 10

    # Save model
    args.save_model = False
    args.save_benchmark = True

    # Composition of arguments given
    args.path = f"./data/preproc/{args.file_name}"
    args.scaler = {"std":StandardScaler()}[args.scaler_name]

    for i in range(number_of_runs):
        args.train_params.run = i
        pipeline(args)
        
