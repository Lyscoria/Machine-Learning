import pandas as pd
from graphviz import Digraph

def split_data(data, column):
    ''' 
    Split the dataset based on the specified feature column.

    Args:
        data (pd.DataFrame): The dataset where the last column is the label, and the other columns are features.
        column (int): The index of the feature column to split on.

    Returns:
        splt_datas (pd.Series): A series where each entry is a subset of the dataset, split by unique values of the specified feature column.
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



def plot_tree(tree, parent_name, node_id=0, graph=None, edge_label=''):
    ''' 
    Recursively plot the decision tree using graphviz, adding nodes and edges for each split.
    
    Args:
        tree (dict): A nested dictionary representing the decision tree.
        parent_name (str): Name of the parent node.
        node_id (int): The identifier for the current node.
        graph (Digraph): Graphviz Digraph object used to build the tree.
        edge_label (str): Label for the edge connecting the parent to the current node.

    Returns:
        int: Updated node_id after adding nodes and edges for the current branch.
    '''
    
    # Initialize the graph only once, at the root of the tree
    if graph is None:
        graph = Digraph(comment='Decision Tree')

    # Base case: If tree node is a leaf (not a dictionary), add a terminal node
    if not isinstance(tree, dict):
        current_node_name = f'node{node_id}' 
        graph.node(current_node_name, label=str(tree)) 
        graph.edge(parent_name, current_node_name, label=edge_label) 
        node_id += 1 
        return node_id  

    # Recursive case: Iterate through the branches (key-value pairs) in the dictionary
    for k, v in tree.items():
        current_node_name = f'node{node_id}'  
        node_label = f'{k}' if isinstance(v, (str, int)) else k  
        graph.node(current_node_name, label=node_label)  
        graph.edge(parent_name, current_node_name, label=str(edge_label))  

        # If the branch is a subtree (dictionary), recursively plot its branches
        if isinstance(v, dict):
            for key in v:
                # Assume the branches can be distinguished with '0' and '1' labels
                node_id += 1  # Increment node_id for each branch
                node_id = plot_tree(v[key], current_node_name, node_id, graph, edge_label=str(key))
                
    return node_id  



def classify(tree, test_data):
    ''' 
    Recursively classify a test instance based on the given decision tree.
    
    Args:
        tree (dict): The decision tree represented as a nested dictionary.
        test_data (Series): The test data instance to classify.

    Returns:
        class_label: The predicted class label for the test instance.
    '''
    
    # Retrieve the first feature name in the current subtree (root of the subtree)
    first_str = list(tree.keys())[0]  
    second_dict = tree[first_str]  
    
    # Find the index of the feature in test_data and retrieve its value
    feat_index = test_data.index.get_loc(first_str)  
    key = test_data.iloc[feat_index]
    
    # Retrieve the value of the subtree for the given test_data value
    value_of_feat = second_dict[key]
    
    # If the result is still a subtree, recursively classify; otherwise, return the class label
    if isinstance(value_of_feat, dict):
        class_label = classify(value_of_feat, test_data)
    else:
        class_label = value_of_feat
    
    return class_label  # Return the final predicted class label

def test(test_data, tree):
    ''' 
    Evaluate the accuracy of the decision tree on the test dataset.
    
    Args:
        test_data (pd.DataFrame): The test dataset containing features and actual labels.
        tree (dict): The decision tree used for classification.

    Returns:
        acc (float): The accuracy of the decision tree on the test dataset.
    '''
    
    # Initialize a list to store 1 if the prediction is correct and 0 otherwise
    class_label = []
    
    # Iterate over each instance in the test dataset
    for i in range(len(test_data)):
        sample = test_data.iloc[i]  # Get each test sample
        # Compare the predicted label with the actual label
        if classify(tree, sample) == sample[-1]:
            class_label.append(1)  # Correct prediction
        else:
            class_label.append(0)  # Incorrect prediction
    
    # Calculate accuracy by dividing the sum of correct predictions by total instances
    acc = sum(class_label) / len(class_label)
    return acc  # Return the accuracy as a fraction (0 to 1)