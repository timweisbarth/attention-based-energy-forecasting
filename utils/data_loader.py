import pandas as pd

def load_data(path):
    """
    Load data from the specified path

    Parameters:
    -----------
    path: string
        The path from which the csv will be loaded
    
    Returns:
    --------
    pd.DataFrame
        The data in a DataFrame format
    """

    df = pd.read_csv(path)

    return df