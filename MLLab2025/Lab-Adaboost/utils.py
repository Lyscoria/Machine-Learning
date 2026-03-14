import pandas as pd


def split_data(data, column):
    ''' 
    Split the dataset based on the specified feature column.

    Args:
        data (DataFrame): The dataset where the last column is the label, and the other columns are features.
        column (int): The index of the feature column to split on.

    Returns:
        Series: A series where each entry is a subset of the dataset, split by unique values of the specified feature column.
    '''
    
    # Step 1: Initialize an empty Series to store the split subsets of the dataset
    splt_datas = pd.Series()  
    
    # Step 2: Retrieve unique values from the specified feature column to use for splitting
    str_values = data.iloc[:, column].unique()  
    
    # Step 3: Loop over unique values, create a subset for each, and add it to splt_datas
    for i in range(len(str_values)):   
        df = data.loc[data.iloc[:, column] == str_values[i]]  # Filter rows matching the current unique value
        splt_datas[str(i)] = df  # Store the filtered subset in the Series
    
    return splt_datas  # Return the Series containing all split subsets