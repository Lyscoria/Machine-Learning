import numpy as np
import unittest
import pandas as pd
from codes import *
from vis_util import *
from data_generator import *

class TestEuclideanDistance(unittest.TestCase):
    def setUp(self):
        self.x1 = np.array([1, 2, 3])
        self.x2 = np.array([4, 5, 6])
        self.x3 = np.array([1, 2, 3])
    
    def test_euclidean_distance(self):
        # Test case 1
        distance = euclidean_distance(self.x1, self.x2)
        self.assertAlmostEqual(distance, 5.196152422706632, places=4, msg="Euclidean distance is incorrect")
        
        # Test case 2
        distance = euclidean_distance(self.x1, self.x3)
        self.assertAlmostEqual(distance, 0.0, places=4, msg="The same point should have a distance of 0")

class TestKNNPredict(unittest.TestCase):
    def setUp(self):
        self.x_train = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        self.y_train = np.array([0, 1, 0, 1])
        self.x_test = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        self.k = 3
    
    def test_KNN_predict(self):
        predictions = KNN_predict(self.x_train, self.y_train, self.x_test, self.k)
        # self.assertListEqual(predictions, [0, 0, 1, 1], msg="Predictions are incorrect")
        self.assertTrue(np.all(predictions == np.array([0, 0, 1, 1])), msg="Predictions are incorrect")

class TestKNNFinal(unittest.TestCase):
    def setUp(self):
        # define the centers of different classes
        centers = [[1, 1], [-1, -1], [1, -1]]
        # define the number of samples
        n_samples = 500
        cluster_std = 0.8
        self.x_train, self.y_train = gen_2D_dataset(centers, n_samples, cluster_std)
        self.x_test, self.y_test = gen_2D_dataset(centers, n_samples, cluster_std)

    def test_KNN_final(self):
        predictions = KNN_predict(self.x_train, self.y_train, self.x_test, 5)
        accuracy = np.mean(predictions == self.y_test)
        self.assertGreaterEqual(accuracy, 0.85, msg="The accuracy is less than the required threshold of 85%")

def run_dist_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEuclideanDistance)
    result = unittest.TextTestRunner().run(suite)

    score = (result.testsRun - (len(result.failures) + len(result.errors))) * 3
    print(f"Final Score of Euclidean Distance: {score}/3")

def run_KNN_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKNNPredict)
    result = unittest.TextTestRunner().run(suite)

    score = (result.testsRun - (len(result.failures) + len(result.errors))) * 2
    print(f"Final Score of KNN Predict: {score}/2")

def run_final_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKNNFinal)
    result = unittest.TextTestRunner().run(suite)

    score = (result.testsRun - (len(result.failures) + len(result.errors))) * 5
    print(f"Final Score of KNN Final: {score}/5")

if __name__ == '__main__':
    unittest.main()