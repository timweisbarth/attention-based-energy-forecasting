from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np


def calc_metrics(preds, truths):
    """
    Calculate the MAE and RMSE of the prediction.

    Parameters:
    -----------
    preds: ndarray
        Predictions of shape (number of samples, prediction_horizon) 

    truths: ndarray
        Ground truth of shape (number of samples, prediction_horizon) 

    Returns:
    --------
    tuple of floats
        The value of the MAE and RMSE
    """

    mae = mean_absolute_error(y_true=truths,
                              y_pred=preds)
    
    mse = mean_squared_error(y_true=truths,
                          y_pred=preds,
                            squared=True)

    return mae, mse

def inverse_transformations(preds, truths, scaler, h):
    """
    Apply the inverse transform to obtain the data without scaling

    Parameters:
    -----------
    preds: ndarray
        Predictions of shape (number of samples, prediction_horizon) 

    truths: ndarray
        Ground truth of shape (number of samples, prediction_horizon) 

    Returns:
    --------
    tuple of ndarrays
        Originally scaled predictions and truths
    """

    #preds = scaler.inverse_transform(preds)
    truths = truths.to_numpy()
    preds_inv = np.copy(preds)
    truths_inv = np.copy(truths)
    
    
    for i in range(h):
        
        preds_inv[:,i::h] = scaler.inverse_transform(preds[:,i::h])
        
        truths_inv[:,i::h] = scaler.inverse_transform(truths[:,i::h])
    #truths_inv = scaler.inverse_transform(truths)
    
    return preds_inv, truths_inv

def print_and_save_benchmark_table(metrics, args):
    """
    Prints the benchmark table to the console. If args.save_benchmark = True, 
    it is also saved to the models folder.

    Parameters:
    -----------
    metrics: dict
        Contains the target variable, the horizon, the mae and rmse
    args: dotdict
        Contains all the arguments of the current run of the pipeline
    """

    # Set a MultiIndex of 'target' and 'horizon'
    metrics_df = pd.DataFrame(metrics)
    metrics_df.set_index(['target', 'horizon'], inplace=True)

    print(metrics_df)

    
