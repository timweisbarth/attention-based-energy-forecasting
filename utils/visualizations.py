import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_prediction_vs_truths(preds, truths, window_size, forecast_horizon, 
                              cols_to_pred, index, plot_date, days, stride, folder_path):
    """Plot the truths vs predictions"""
    
    # Select the first column if in multivariate setting
    if len(cols_to_pred) > 1:
        
        truths = truths[:,:forecast_horizon]
        preds = preds[:,:forecast_horizon]
        
    # Select rows, if the forecast_horizon is larger than the stride. 
    if forecast_horizon > stride:
        truths = truths[::int(forecast_horizon/stride)]
        preds = preds[::int(forecast_horizon/stride)]
    
    
    preds = preds.flatten()
    truths = truths.flatten()

    # Slice index such that it aligns with truths and preds
    index = index[:preds.shape[0]]

    # Get index of the plot_date and index of "days" later days
    length_of_interval = 24 * days
    
    
    start = np.where(index == plot_date)[0][0]
    stop = start + length_of_interval

    # Plotting
    plt.figure(figsize=(15,6))
    plt.plot(index[start:stop], truths[start:stop], label="Actual Values", color='blue', marker='o')
    plt.plot(index[start:stop], preds[start:stop], label="Predicted Values", color='red', linestyle='dashed', marker='x')
    plt.title(f'Truth vs Prediction for {cols_to_pred[0]}, window size: {window_size}, forecast horizon: {forecast_horizon}')
    plt.ylabel('Generation Value')
    plt.xlabel('Time Step')
    plt.legend()
    plt.grid(True)
    plt.savefig(folder_path + "{}_{}.pdf".format(plot_date, days), format="pdf", bbox_inches="tight") 

    np.save(folder_path + '_pred.npy', preds[start:stop])
    np.save(folder_path + '_true.npy', truths[start:stop])
    #plt.show()