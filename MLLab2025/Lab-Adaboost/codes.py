import numpy as np
import pandas as pd

class WeakClassifier:
    def __init__(self):
        self.tree = None 
        self.alpha = None
    
    def best_split(self, X, y, sample_weight):
        ''' 
        Find the optimal feature and threshold for splitting the data using Gini impurity.

        Args:
            X (pd.DataFrame): Data features, shape (n_samples, n_features).
            y (pd.Series): Data labels, shape (n_samples,).
            sample_weight (np.ndarray): Sample weights, shape (n_samples,).

        Returns:
            best_feature (str): Name of the feature selected for the split.
            best_threshold (float or int): Optimal threshold for the split.
            best_splits (tuple): Tuple containing masks for left and right splits (np.ndarray, np.ndarray).
        '''
        # TODO: Implement the function to find the best feature, threshold and splits to split the data based on the Gini impurity.

        def calc_gini(y, w):
            tot_w = np.sum(w)
            if tot_w == 0:
                return 0
            pos_w = np.sum(w[y == 1])
            neg_w = np.sum(w[y == -1])
            gini = 1 - (pos_w / tot_w) ** 2 - (neg_w / tot_w) ** 2
            return gini
        
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        best_splits = None
        
        for feature in X.columns:
            thresholds = X[feature].unique()
            for threshold in thresholds:
                lmask = (X[feature] <= threshold).values
                rmask = ~lmask
                if np.sum(lmask) == 0 or np.sum(rmask) == 0:
                    continue
                w_l = sample_weight[lmask]
                y_l = y[lmask]
                w_r = sample_weight[rmask]
                y_r = y[rmask]

                gini = np.sum(w_l) / np.sum(sample_weight) * calc_gini(y_l, w_l) + \
                    np.sum(w_r) / np.sum(sample_weight) * calc_gini(y_r, w_r)
                
                if gini < best_gini:
                    best_gini = gini
                    best_threshold = threshold
                    best_feature = feature
                    best_splits = (lmask, rmask)
        
        return best_feature, best_threshold, best_splits
        
        
    def fit(self, X, y, sample_weight):
        '''  
        Train the weak classifier on the data.

        Args:
            X (pd.DataFrame): Data features, shape (n_samples, n_features).
            y (pd.Series): Data labels, shape (n_samples,).
            sample_weight (np.ndarray): Sample weights, shape (n_samples,).

        Returns:
            None: Updates self.tree with the trained decision tree structure.
        '''
        best_feature, best_threshold, best_splits = self.best_split(X, y, sample_weight)
        # TODO: Create the tree as a nested dictionary
        lmask, rmask = best_splits

        def calc_majority(y, w):
            pos_w = np.sum(w[y == 1])
            neg_w = np.sum(w[y == -1])
            if pos_w >= neg_w:
                return 1
            else:
                return -1
            
        lpred = calc_majority(y[lmask], sample_weight[lmask])
        rpred = calc_majority(y[rmask], sample_weight[rmask])

        self.tree = {
            'feature': best_feature,
            'threshold': best_threshold,
            'left_pred': lpred,
            'right_pred': rpred
        }

    def predict(self, x):
        '''  
        Predict labels for the given data.

        Args:
            x (pd.DataFrame): Data features for prediction, shape (n_samples, n_features).

        Returns:
            predict_labels (np.ndarray): Predicted labels, shape (n_samples,).
        '''

        # Initialize list to store prediction results
        predict_labels = []

        # Predict label for each sample
        for i in range(len(x)):
            sample = x.iloc[i,:]

            # TODO: Predict the label of the sample
            if sample[self.tree['feature']] <= self.tree['threshold']:
                predict_labels.append(self.tree['left_pred'])
            else:
                predict_labels.append(self.tree['right_pred'])
        
        return np.array(predict_labels)
    
    
class Adaboost:
    def __init__(self, n_estimators=10):
        '''
        Initialize the Adaboost classifier.

        Args:
            n_estimators (int): Number of weak classifiers in the ensemble.
        '''
        # Number of weak classifiers to use
        self.n_estimators = n_estimators
        # List to store each weak classifier
        self.clfs = []
    
    # AdaBoost training process
    def fit(self, X, y):
        '''  
        Train the Adaboost ensemble.

        Args:
            X (pd.DataFrame): Data features, shape (n_samples, n_features).
            y (pd.Series): Data labels, shape (n_samples,).

        Returns:
            None: Updates the list of weak classifiers (self.clfs).
        '''
        n_samples, m_features = X.shape
    
        # Initialize weights for each sample
        w = np.ones(n_samples) / n_samples

        # Train each weak classifier
        for _ in range(self.n_estimators):
            clf = WeakClassifier()

            # 1. Fit the weak classifier using the current weights
            clf.fit(X, y, w)

            # TODO: 2. Predict the labels of the data using the weak classifier
            y_pred = clf.predict(X)
            
            # TODO: 3. Calculate the error rate by comparing predictions to actual labels
            err = np.sum(w[y != y_pred])

            # TODO: 4. Calculate alpha (the weight of the classifier in the final model)
            alpha = 0.5 * np.log((1 - err) / err)

            # TODO: 5. Update the sample weights to emphasize misclassified samples
            w *= np.exp(-alpha * y * y_pred)
            
            # Normalize weights so that they sum to one
            w /= np.sum(w)

            # Store the classifier and its weight
            clf.alpha = alpha
            self.clfs.append(clf)
            

    def predict(self, X):
        '''  
        Predict the label for each sample using the Adaboost ensemble.

        Args:
            X (pd.DataFrame): Data features for prediction, shape (n_samples, n_features).

        Returns:
            predict_labels (np.ndarray): Predicted labels, shape (n_samples,).
        '''

        # TODO: 1. Compute the predictions of each weak classifier
        # TODO: 2. Compute the weighted sum of predictions from all classifiers
        n = X.shape[0]
        predictions = np.zeros(n)
        for clf in self.clfs:
            predictions += clf.alpha * clf.predict(X)
        
        # TODO: 3. Determine the final label based on the sign of the weighted sum (if x>0 return 1, else return -1)
        predicted_labels = np.where(predictions >= 0, 1, -1)
        
        return predicted_labels