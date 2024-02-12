import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import date
import holidays
from workalendar.europe import Germany
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import timefeatures as tf



def all_preproc_steps(df, cols_to_predict, scaler, w):
    """Apply all preprocessing steps. See docs of individual functions."""

    
    df = preproc0_rename(df)
    df = preproc1_nans(df)
    df = preproc2_time_features(df)
    
    df_train, df_val, df_test = train_val_test_split(df, w)
    df_train, df_val, df_test = scale_data(scaler, df_train, df_val, df_test, cols_to_predict)

    return df_train, df_val, df_test

def preproc0_rename(df):
    """Rename the columns of the DataFrame df"""

    df = df.rename(columns={"DE_solar_generation_actual": "solar_gen", "DE_wind_generation_actual": "wind_gen", "DE_load_actual_entsoe_transparency": "load"})

    return df


def preproc1_nans(df):
    """
    Fill nans of df with varying techniques per column.

    Solar generation: Daily interpolation
    Wind generation: quadratic interpolation
    Load: Only the first value is missing 

    See nan_exploration.ipynb for details
    """

    # Solar column
    if df["solar_gen"].isna().any():
        print("Dealing with NaNs in solar_gen")
        nan_rows_solar = df["solar_gen"][df["solar_gen"].isna()]

        for i in list(nan_rows_solar.index):
            if i < 10:
                df.loc[i, "solar_gen"] = df.loc[i+24, "solar_gen"]
            else:
                df.loc[i, "solar_gen"] = (df.loc[i-24, "solar_gen"] + df.loc[i+24, "solar_gen"]) / 2

    # Wind column
    if df["wind_gen"].isna().any():
        print("Dealing with NaNs in wind_gen")
        df["wind_gen"] = df["wind_gen"].interpolate(limit_area='inside', method="quadratic")  # Interpolate for inside NaNs
        df.loc[0, "wind_gen"] = df.loc[1, "wind_gen"]

    # Load column

    
    if df["load"].isna().any():
        print("Dealing with NaNs in load")
        df.loc[0, "load"] = df.loc[1, "load"]

    return df

def preproc2_time_features(df):
    """
    Introduce time features year, month, day, dayofweek, hour,
    isSaturday, isSunday, isholidayto to df. Make month, dayofweek
    and hour cyclic.
    """

    # Remove unnecessary columns
    #df = df.drop(columns="cet_cest_timestamp")


    
    # Add time features
    #df["date"] = pd.to_datetime(df["date"])

    df = df.set_index(["date"])
    df.index = pd.to_datetime(df.index)
    
    #display(df)

    cols = [
        "HourOfDay_sin", "HourOfDay_cos",
        "DayOfWeek_sin", "DayOfWeek_cos",
        "DayOfYear_sin", "DayOfYear_cos",
        "IsHoliday", "IsWeekend", "year"
    ]

    time_features = pd.DataFrame(
        data=tf.time_features(pd.to_datetime(df.index.values), "h").T,
        index=pd.to_datetime(df.index),
        columns=cols
    )

    time_features["year"] = df.index.year #trend##

    df = pd.concat((df, time_features), axis=1)
    #display(df.head(30))
    #display(df.tail(50))

    return df

def train_val_test_split(df, w):
    """
    Split the df in a Train (2015-2017), Validation (2018) 
    and Test (2019) set

    Parameters:
    -----------
    df: pd.DataFrame
        

    Returns:
    --------
    tuple pd.DataFrame, pd.DataFrame, pd.DataFrame 
        Train, validation and test data
    """
    

    df_train = df[df["year"] < 2021] # 7 years

    # Validation set consists of years 2021 and 2022
    # We can take last w elements from train set without information leakage
    df_val1 = df_train[-w:]
    df_val2 = df[(df["year"] == 2021) | (df["year"] == 2022)] # 2 years
    df_val = pd.concat((df_val1, df_val2), axis=0)

    # Test set consists of year 2023
    # We can take last w elements from val set without information leakage
    df_test1 = df_val[-w:]
    df_test2 = df[df["year"] > 2022] # 1 year
    df_test = pd.concat((df_test1, df_test2), axis=0)

    return df_train, df_val, df_test
    

def scale_data(scaler, df_train, df_val, df_test, col_to_pred):
    """
    Apply a scaler to the features and labels seperately.

    Parameters:
    -----------
    scaler: sklearn.preprocessing
        e.g. StandardScaler()
    df_train, df_val, df_test: pd.DataFrame
        Trainnig, validation and test data
    col_to_pred:
        The columns for which a prediction is needed

    Returns:
    -------
    tuple of pd.DataFrames
        The scaled train, validation and test data

    """
    
    # Split dfs such that features are seperated from labels
    y_train = df_train[col_to_pred]
    y_val = df_val[col_to_pred]
    y_test = df_test[col_to_pred]
    #print(y_train.shape)

    X_train = df_train.drop(columns=col_to_pred, axis=1)
    X_val = df_val.drop(col_to_pred, axis=1)
    X_test = df_test.drop(col_to_pred, axis=1)
    #print(X_train.shape)


    # Apply scaling on features and labels seperately. This is useful later
    # for the inverse scaling
    X_train_arr = scaler.fit_transform(X_train)
    X_val_arr = scaler.transform(X_val)
    X_test_arr = scaler.transform(X_test)

    y_train_arr = scaler.fit_transform(y_train)
    y_val_arr = scaler.transform(y_val)
    y_test_arr = scaler.transform(y_test)

    # Rebuild dfs from ndarrays
    X_train = pd.DataFrame(X_train_arr, columns=X_train.columns, index=X_train.index)
    y_train = pd.DataFrame(y_train_arr, columns=y_train.columns, index=y_train.index)
    X_val = pd.DataFrame(X_val_arr, columns=X_val.columns, index=X_val.index)
    y_val = pd.DataFrame(y_val_arr, columns=y_val.columns, index=y_val.index)
    X_test = pd.DataFrame(X_test_arr, columns=X_test.columns, index=X_test.index)
    y_test = pd.DataFrame(y_test_arr, columns=y_test.columns, index=y_test.index)
    df_train = pd.concat([X_train, y_train], axis=1)
    df_val = pd.concat([X_val, y_val], axis=1)  
    df_test = pd.concat([X_test, y_test], axis=1)  
    #print(df_train.shape)  

    return df_train, df_val, df_test


def create_inout_sequence(df_train, df_val, df_test, target, h, w, stride):
    """ 
    Creates an inout sequence for the nn.lstm module. 
    
    Parameters:
    -----------
    df_{train, val, test}: pd.DataFrame
        train, val and test set
    target: string
        The name of the time-series that will be forecasted
    h: int
        forecast horizon
    w: int
        window_size
    stride: int
        The amount by which the data is shifted after the generation 
        of each inout sequence

    Returns:
    --------
    [(ndarray1, ndarray2), (ndarray1, ndarray2), (ndarray1, ndarray2)]:
        Each tuple contains the in and out sequence of the train, val or test set.
        ndarray1 is of shape (n_samples, w, input_size) and ndarray2 is of shape
        (n_smaples, h) where n_samples depends on the size of the respective set
        This is according to the nn.lstm documentation.
    """
    seq_length = w + h
    
    inout_seqs = []
    #display(df_train["load"][23:56])
    for df in [df_train.copy(), df_val.copy(), df_test.copy()]:
        length = df.shape[0]
        in_seqs = np.array([])
        out_seqs = np.array([])
        
        for i in range(0, length-seq_length+1, stride):
            
            in_seq = df.iloc[i:i+w].values
            #if i == 0:
                #print(in_seq[0:5, -1])
            out_seq = df[target].iloc[i+w:i+w+h].values

            in_seq = in_seq[np.newaxis, :, :]
            out_seq = out_seq[np.newaxis, :]

            in_seqs = np.vstack((in_seqs, in_seq)) if in_seqs.size > 0 else in_seq
            out_seqs = np.vstack((out_seqs, out_seq)) if out_seqs.size > 0 else out_seq
        
        inout_seqs.append((in_seqs, out_seqs))
    return inout_seqs


def make_supervised(df_train, df_val, df_test, targets, h, w, stride, cols_to_lag):
    """
    Turn the time series in a supervised problem, i.e. each df (train,val,test) is
    split in a df of features (including lags) and targets.

     Parameters:
    -----------
    df_{train, val, test}: pd.DataFrame
        train, val and test set
    targets: list of strings
        The names of the time-series that will be forecasted
    h: int
        forecast horizon
    w: int
        window_size
    stride: int
        The amount by which the data is shifted after the generation 
        of each inout sequence
    cols_to_lag: list of strings
        The name of the columns that should be lagged

    Returns:
    --------
    (pd.DataFrame1, pd.DataFrame2), (pd.DataFrame1, pd.DataFrame2), (pd.DataFrame1, pd.DataFrame2)
        Each tuple contains the in and out sequence of the train, val or test set.
        pd.DataFrame1 is of shape (n_samples, n_features) and pd.DataFrame2 is of shape
        (n_smaples, h) where n_samples depends on the size of the respective set and
        n_features depends on the number of columns of the input df, the window size and
        len(cols_to_lag).
    """
    dfs = []

    if len(targets) == 1:
            cols_to_lag = targets

    # The new shape will be (n_samples_old - w - h + 1), +1 because only w-1 lags are produced due to current lag
    for df in [df_train.copy(), df_val.copy(), df_test.copy()]:
        
        lagged_columns = {}
        shifted_columns = {}


        
        for col in cols_to_lag:
            # get window_size-1 many lags --> -1 due to current lag
            for lag in range(1,w):
                #print(lag, end=", ")
                lagged_columns[f"{col}_lag{lag}"] = df[f"{col}"].shift(lag)

        for col in targets:
            for shift in range(1,h+1):
                shifted_columns[f"{col}_next_{shift}"] = df[col].shift(-shift)

        # Concatenate lagged and shifted columns to the DataFrame
        df = pd.concat([df, pd.DataFrame(lagged_columns), pd.DataFrame(shifted_columns)], axis=1)
        df.dropna(axis=0, inplace=True)     

        # Split df in X,y at the forward shiftet locations
        col_index = df.columns.get_loc(f"{targets[0]}_next_1")
        X = df.iloc[:, :col_index]
        y = df.iloc[:, col_index:]

        #display(X)
        #display(y)
        
        X = X[::stride]
        y = y[::stride]
        

        dfs.append((X,y))

    return dfs[0], dfs[1], dfs[2]

def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    """Create the data loaders from the train, val and test data"""

    train_features = torch.Tensor(X_train)
    train_targets = torch.Tensor(y_train)
    val_features = torch.Tensor(X_val)
    val_targets = torch.Tensor(y_val)
    test_features = torch.Tensor(X_test)
    test_targets = torch.Tensor(y_test)

    train = TensorDataset(train_features, train_targets)
    val = TensorDataset(val_features, val_targets)
    test = TensorDataset(test_features, test_targets)


    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader_one = DataLoader(val, batch_size=1, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader, val_loader_one