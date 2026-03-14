import numpy as np
import unittest
import pandas as pd
from codes import *
from utils import *
import unittest
import pandas as pd
import numpy as np

class TestGetInfoEntropy(unittest.TestCase):
    def setUp(self):
        # Create example datasets with known entropy values for testing
        self.data_uniform = pd.DataFrame({'feature1': [1, 2, 3, 4], 'label': [1, 1, 1, 1]})  # All labels the same
        self.data_binary = pd.DataFrame({'feature1': [1, 2, 3, 4], 'label': [1, 0, 1, 0]})   # 50/50 split in labels
        self.data_mixed = pd.DataFrame({'feature1': [1, 2, 3, 4, 5, 6], 'label': [1, 0, 1, 0, 1, 1]})  # Mixed labels
        
    def test_getInfoEntropy(self):
        # Expected entropy for a uniform dataset (all labels the same) is 0
        entropy = getInfoEntropy(self.data_uniform)
        self.assertAlmostEqual(entropy, 0.0, places=4, msg="Entropy should be 0 for uniform label dataset")
        
        entropy = getInfoEntropy(self.data_binary)
        self.assertAlmostEqual(entropy, 1.0, places=4, msg="Entropy should be 1 for binary split dataset")
        
        p1 = 4/6  # Probability of label 1
        p0 = 2/6  # Probability of label 0
        expected_entropy = - (p1 * np.log2(p1) + p0 * np.log2(p0))
        
        entropy = getInfoEntropy(self.data_mixed)
        self.assertAlmostEqual(entropy, expected_entropy, places=4, msg="Entropy should match expected for mixed label dataset")

class TestFindBestFeature(unittest.TestCase):
    
    def setUp(self):
        # Dataset with a clear best feature based on label separation
        self.data_1 = pd.DataFrame({
            'feature1': [1, 1, 0, 0, 1, 0],
            'feature2': [0, 1, 1, 1, 0, 1],
            'label': [1, 1, 0, 0, 1, 0]
        })
        
        self.data_2 = pd.DataFrame({
            'feature1': [0, 1, 0, 0, 1, 0, 1, 0],
            'feature2': [1, 0, 1, 0, 1, 0, 1, 0],
            'feature3': [2, 2, 2, 2, 2, 2, 2, 2],  # Irrelevant feature (constant)
            'feature4': [1, 0, 0, 1, 1, 0, 1, 0],
            'label':    [1, 1, 0, 0, 1, 0, 1, 0]
        })

        # Dataset where labels correlate with a combination of features
        self.data_3 = pd.DataFrame({
            'feature1': [1, 1, 0, 0, 1, 0, 1, 0],
            'feature2': [1, 1, 1, 0, 1, 1, 0, 0],
            'feature3': [1, 1, 1, 1, 1, 0, 0, 0],
            'feature4': [1, 0, 0, 1, 0, 1, 1, 0],
            'label':    [1, 1, 0, 0, 1, 0, 1, 0]
        })

    def test_find_best_feature_data_1(self):
        best_feature, best_series = find_best_feature(self.data_1)
        expected_best_feature = 'feature1'  # Assume feature1 is expected
        self.assertEqual(best_feature, expected_best_feature, 
                         f"Expected best feature: {expected_best_feature}, but got: {best_feature}")
        self.assertIsInstance(best_series, pd.Series)
        self.assertEqual(len(best_series), len(self.data_1[best_feature].unique()))

    def test_find_best_feature_data_2(self):
        best_feature, best_series = find_best_feature(self.data_2)
        expected_best_feature = 'feature1'  # Assume feature1 is expected
        self.assertEqual(best_feature, expected_best_feature, 
                         f"Expected best feature: {expected_best_feature}, but got: {best_feature}")
        self.assertIsInstance(best_series, pd.Series)
        self.assertEqual(len(best_series), len(self.data_1[best_feature].unique()))

    def test_find_best_feature_data_3(self):
        best_feature, best_series = find_best_feature(self.data_3)
        expected_best_feature = 'feature1'  # feature2 is irrelevant as it has the same value
        self.assertEqual(best_feature, expected_best_feature, 
                         f"Expected best feature: {expected_best_feature}, but got: {best_feature}")
        self.assertIsInstance(best_series, pd.Series)
        self.assertEqual(len(best_series), len(self.data_3[best_feature].unique()))

class TestTree(unittest.TestCase):
    def setUp(self):
        # Load training and test data from CSV files
        self.train_data = pd.read_csv('train_phd_data.csv')
        self.test_data = pd.read_csv('test_phd_data.csv')


    def testTreeAcc(self):
        # Generate the tree model from training data
        Tree = create_Tree(self.train_data)
        
        # Test accuracy on test and training datasets
        acctest = test(self.test_data, Tree)
        acctrain = test(self.train_data, Tree)
        
        
        # Allow for a slight tolerance in the comparison
        self.assertGreater(acctrain, 0.9)
        self.assertGreater(acctest, 0.8)

# Function to run TestGetInfoEntropy tests and calculate the score
def run_get_info_entropy_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGetInfoEntropy)
    result = unittest.TextTestRunner().run(suite)
    score = (result.testsRun - (len(result.failures) + len(result.errors))) * 3.3333333
    print(f"Final Score of Info Entropy: {score}/3.33")

# Function to run TestFindBestFeature tests and calculate the score
def run_find_best_feature_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFindBestFeature)
    result = unittest.TextTestRunner().run(suite)
    score = (result.testsRun - (len(result.failures) + len(result.errors))) * 1.1111111
    print(f"Final Score of Find Best Feature: {score}/3.33")

# Function to run TestTree tests and calculate the score
def run_tree_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTree)
    result = unittest.TextTestRunner().run(suite)
    score = (result.testsRun - (len(result.failures) + len(result.errors))) * 3.333333
    print(f"Final Score of Tree: {score}/3.33")


if __name__ == '__main__':
    unittest.main()