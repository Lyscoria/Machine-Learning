import numpy as np
import unittest
import pandas as pd
from codes import *
from utils import *
import unittest
import pandas as pd
import numpy as np

class TestWeakClassifier(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1]
        })
        self.y = pd.Series([1, 1, -1, -1, 1])
        
        self.sample_weight = np.array([1, 1, 1, 1, 1])
        self.classifier = WeakClassifier()

    def test_best_split(self):
        # Test the best_split function
        best_feature, best_threshold, best_splits = self.classifier.best_split(self.X, self.y, self.sample_weight)

        # Assert the best feature and threshold are valid
        self.assertIn(best_feature, self.X.columns, "The best feature should be a valid column name.")
        self.assertIsNotNone(best_threshold, "The best threshold should not be None.")
        self.assertIsInstance(best_splits, tuple, "Splits should be returned as a tuple.")
        self.assertEqual(len(best_splits), 2, "Splits tuple should contain two elements (left and right masks).")



    def test_fit(self):
        # Test if the fit method correctly sets up the tree structure
        self.classifier.fit(self.X, self.y, self.sample_weight)
        
        # Check if tree has been created after fitting
        self.assertIsNotNone(self.classifier.tree, "The tree after fitting should not be None.")

        
        
        
    def test_predict(self):
        # Fit the model first
        self.classifier.fit(self.X, self.y, self.sample_weight)
        
        # Predict using the fitted model
        predictions = self.classifier.predict(self.X)
        self.assertAlmostEqual(predictions.shape, self.y.shape, "Prediction should have the same shape as self.y.")
        
    
class TestAdaboost(unittest.TestCase):
    def setUp(self):
        # Sample data setup
        self.X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1]
        })
        self.y = pd.Series([1, 1, -1, -1, 1])
        
        # Initialize the Adaboost model with a few estimators
        self.model = Adaboost(n_estimators=5)
    
    def test_fit(self):
        # Test if the fit method trains and stores the weak classifiers correctly
        self.model.fit(self.X, self.y)
        
        # Ensure each classifier has a valid alpha weight assigned
        for clf in self.model.clfs:
            self.assertIsInstance(clf.alpha, float, "Each weak classifier should have an alpha weight of type float.")
            self.assertIsNotNone(clf.tree, "The tree after fitting should not be None.")
        self.assertAlmostEqual(len(self.model.clfs), self.model.n_estimators)
            
class TestAcc(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        
    def test_fit(self):
        
        train_data = pd.read_csv('train_phd_data.csv')
        test_data = pd.read_csv('test_phd_data.csv')

        train_data.iloc[:, -1] = train_data.iloc[:, -1].map({0: -1, 1: 1})
        test_data.iloc[:, -1] = test_data.iloc[:, -1].map({0: -1, 1: 1})
        self.adaboost_model = Adaboost(n_estimators=10)
        self.adaboost_model.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])

        predicted = self.adaboost_model.predict(test_data.iloc[:, :-1])
        accuracy = np.sum(predicted == test_data.iloc[:, -1])/len(test_data)
        
        self.assertGreater(accuracy, 0.9, "Make sure your Adaboost model can achieve accuracy larger than 90%.")
        

def run_weak_classifier_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestWeakClassifier)
    result = unittest.TextTestRunner().run(suite)
    
    # Calculate score: 10 points max, scaled based on successful tests
    score = (result.testsRun - (len(result.failures) + len(result.errors))) * 3 / result.testsRun
    print(f"Final Score of Weak Classifier: {score}/3")
    

def run_adaboost_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAdaboost)
    result = unittest.TextTestRunner().run(suite)
    
    # Calculate score: 10 points max, scaled based on successful tests
    score = (result.testsRun - (len(result.failures) + len(result.errors))) * 3 / result.testsRun
    print(f"Final Score of Adaboost: {score}/3")
    
def run_acc_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAcc)
    result = unittest.TextTestRunner().run(suite)
    
    # Calculate score: 10 points max, scaled based on successful tests
    score = (result.testsRun - (len(result.failures) + len(result.errors))) * 6 / result.testsRun
    print(f"Final Score of Adaboost: {score}/6")
    
    