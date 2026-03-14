import numpy as np
import pandas as pd
from utils import *

def getInfoEntropy(data):
    ''' 
    Calculate the information entropy of the dataset.

    Args:
        data (pd.DataFrame): Dataset where the last column represents labels, and the other columns represent features.

    Returns:
        Entropy (float): The information entropy of the data.
        
    Note: Use log2 for log calculation.
    '''

    # TODO: Step 1. Count the number of samples for each label (class) and store in count_class
    # Hint: Use pd.value_counts() to get the count of samples per label.
    #       Use data.iloc[:, -1] to access the last column of the dataset.
    count_class = data.iloc[:, -1].value_counts()
    
    # TODO: Step 2. Calculate the total number of samples in the dataset
    data_count = data.shape[0]
    
    Entropy = 0.0
    # TODO: Step 3. Loop through each class to calculate its probability and entropy
    for i in range(len(count_class)):
        # TODO: Step 3a. Calculate the probability of each class
        p = count_class.iloc[i] / data_count
        
        # TODO: Step 3b. Calculate and add the entropy contribution of each class to Entropy
        Entropy += -p * np.log2(p)
    
    return Entropy


def find_best_feature(data):
    '''  
    Identify the best feature to split the dataset based on information entropy gain ratio.

    Args:
        data (pd.DataFrame): The dataset where the last column is the label, and the other columns are features.

    Returns:
        best_feature (str): The name of the best feature for splitting.
        best_Series (pd.Series): The dataset split by the best feature, outputed by function `split_data`.
    '''
    
    best_feature_index = 0    
    baseEnt = getInfoEntropy(data)  
    bestInfoGain_ratio = 0.0
    numFeatures = data.shape[1] - 1   
    InfoGain = 0.0 
    
    # Iterate over each feature to calculate its information gain ratio
    for i in range(numFeatures):
        newEnt = 0.0
        # Small constant to avoid division by zero
        IV = 1e-5
        
        # TODO: Step 1: Split the dataset according to the current feature column
        series = split_data(data, i)

        # TODO: Step 2: Calculate the information entropy and weighted average information entropy for each subset
        for j in range(len(series)):
            df = series.iloc[j]
            
            # TODO: Step 3: Calculate the probability of each subset
            p = len(df) / len(data)
            
            # TODO: Step 4: Accumulate the weighted average information entropy
            newEnt += p * getInfoEntropy(df)
            
            # TODO: Step 5: Calculate the intrinsic value (IV) for the feature
            IV += -p * np.log2(p)
        
        # TODO: Step 6: Calculate the information gain for the current feature
        InfoGain = baseEnt - newEnt
        
        # TODO: Step 7: Calculate the information gain ratio for the current feature
        InfoGain_ratio = InfoGain / IV

        # Update the best feature if the current feature has a higher information gain ratio
        if InfoGain_ratio > bestInfoGain_ratio:
            bestInfoGain_ratio = InfoGain_ratio
            best_feature_index = i
            best_Series = series
        
    return data.columns[best_feature_index], best_Series


def create_Tree(data):

    '''
    Build a decision tree from the dataset.

    Args:
        data (pd.DataFrame): The dataset where the last column is the label, and the other columns are features.

    Returns:
        Tree (dict): A dictionary representing the decision tree.
    '''
    
    # Retrieve the unique class labels in the dataset
    y_values = data.iloc[:, -1].unique()   

    # TODO: Step 1: Implement stopping condition for a single class label: If there is only one class label, stop splitting and return this label
    if len(y_values) == 1:
        return y_values[0]
    
    # Step 2: Check if each feature has the same value across all samples. If so, return the class label with the most occurrences
    flag = 0
    for i in range(data.shape[1] - 1):   
        if len(data.iloc[:, i].unique()) != 1:
            flag = 1
            break
    
    
    # TODO: Step 3: If all features are identical, return the class label with the highest frequency
    if (flag == 0):
        value_count = data.iloc[:, -1].value_counts()
        return value_count.idxmax()

    # TODO: Step 4: Find the best feature to split the dataset
    best_feature, best_Series = find_best_feature(data)
    
    Tree = {best_feature:{}}
    # TODO: Step 5: Recursively build the tree for each subset created by splitting on the best feature
    for j in range(len(best_Series)):    
        split_data = best_Series.iloc[j]
        
        # Extract the unique value of the best feature for the current split
        value = split_data.loc[:, best_feature].unique()[0]  

        # Remove the best feature column from the current split data
        split_data = split_data.drop(best_feature, axis=1) 
        
        # TODO: Step 6: Recursively call the function to build subtrees for each split
        Tree[best_feature][value] = create_Tree(split_data)
    
    return Tree