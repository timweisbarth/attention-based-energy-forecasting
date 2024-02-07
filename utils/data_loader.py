import pandas as pd

def load_data(path, from_raw):
    """
    Load data from the specified path

    Parameters:
    -----------
    path: string
        The path from which the csv will be loaded
    from_raw: bool
        Load data from raw?
    
    Returns:
    --------
    pd.DataFrame
        The data in a DataFrame format
    """
    if from_raw:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index)

    return df