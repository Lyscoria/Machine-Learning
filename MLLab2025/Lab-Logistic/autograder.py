import numpy as np
import matplotlib.pyplot as plt
from codes import *
from autograder import *
from vis_util import *
from data_generator import *
import unittest

class TestPredict(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[1, 2], [2, 3], [3, 4]])
        self.w = np.array([1, 2, 3])
        self.lr = LogisticRegression()
        self.lr.w = self.w

    def test_predict(self):
        x = np.concatenate((self.x, np.ones((self.x.shape[0], 1))), axis=1) 
        prob, pred = self.lr.predict(x)
        np.testing.assert_array_almost_equal(prob, np.array([0.9996646498695336, 0.999983298578152, 0.9999991684719722]), decimal=8)
        np.testing.assert_array_equal(pred, np.array([1, 1, 1]))

class TestLoss(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[1, 2], [2, 3], [3, 4]])
        self.y = np.array([1, -1, 1])
        self.w = np.array([1, 2, 3])
        self.lr = LogisticRegression()
        self.lr.w = self.w
        self.lr.reg = 0.1

    def test_calLoss(self):
        x = np.concatenate((self.x, np.ones((self.x.shape[0], 1))), axis=1) 
        loss = self.lr.calLoss(x, self.y)
        self.assertAlmostEqual(loss, 11.000352939462587)

    def test_calLossReg(self):
        x = np.concatenate((self.x, np.ones((self.x.shape[0], 1))), axis=1) 
        loss = self.lr.calLossReg(x, self.y)
        self.assertAlmostEqual(loss, 12.400352939462588)

class TestLogistic(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.x_train, self.y_train = gen_2D_dataset(100, 100, noise = 0)
        self.x_test, self.y_test = gen_2D_dataset(50, 50, noise = 0.7) 
        self.lr = LogisticRegression()

    def test_fit(self):
        self.lr.fit(self.x_train, self.y_train, lr=0.1, n_iter=1000, reg=0)
        x = np.concatenate((self.x_train, np.ones((self.x_train.shape[0], 1))), axis=1) 
        prob, pred = self.lr.predict(x)
        acc_train = np.mean(pred == self.y_train)
        self.assertGreaterEqual(acc_train, 0.99)
        prob, pred = self.lr.predict(np.concatenate((self.x_test, np.ones((self.x_test.shape[0], 1))), axis=1))
        acc_test = np.mean(pred == self.y_test)
        self.assertGreaterEqual(acc_test, 0.97)

class TestComputeAcc(unittest.TestCase):
    def setUp(self):
        self.y = np.array([1, -1, 1, 1, -1, 1])
        self.pred = np.array([1, -1, 1, 1, -1, 1])

    def test_compute_acc(self):
        acc = compute_accuracy(self.y, self.pred)
        self.assertEqual(acc, 1)

def run_Logistic_test():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLogistic)
    result = unittest.TextTestRunner().run(suite)

    score = (result.testsRun - (len(result.failures) + len(result.errors))) / result.testsRun * 3
    print(f"Final Score of Logistic test: {score}/3")

def run_predict_test():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPredict)
    result = unittest.TextTestRunner().run(suite)

    score = (result.testsRun - (len(result.failures) + len(result.errors))) / result.testsRun * 2
    print(f"Final Score of predict test: {score}/2")

def run_loss_test():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLoss)
    result = unittest.TextTestRunner().run(suite)

    score = (result.testsRun - (len(result.failures) + len(result.errors))) / result.testsRun * 2
    print(f"Final Score of loss test: {score}/2")

def run_compute_acc_test():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestComputeAcc)
    result = unittest.TextTestRunner().run(suite)

    score = (result.testsRun - (len(result.failures) + len(result.errors))) / result.testsRun * 1
    print(f"Final Score of compute acc test: {score}/1")

if __name__ == '__main__':
    unittest.main()