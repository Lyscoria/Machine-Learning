import numpy as np
import unittest
import pandas as pd
from codes import *
from vis_util import *

class TestClosedFormSolution(unittest.TestCase):
    def setUp(self):
        self.X1 = np.random.rand(100)
        self.y1 = 3 * self.X1 + 2
        self.X1 = self.X1.reshape(-1, 1)

        self.X2 = np.array([[1, 1], [2, 2], [3, 4], [6, 2]])
        self.y2 = np.array([1, 2, 3, 4])

    def test_closed_form_solution(self):
        y_pred, theta = closed_form_solution(self.X1, self.y1)
        self.assertEqual(y_pred.shape, self.y1.shape)
        self.assertEqual(theta.shape, (2, ))

        y_pred, theta = closed_form_solution(self.X2, self.y2)
        self.assertEqual(y_pred.shape, self.y2.shape)
        self.assertEqual(theta.shape, (3, ))

class TestPredict(unittest.TestCase):
    def setUp(self):
        self.X1 = np.random.rand(100)
        self.y1 = 3 * self.X1 + 2
        self.X1 = self.X1.reshape(-1, 1)

        self.X2 = np.array([[1, 1], [2, 2], [3, 4], [6, 2]])
        self.y2 = np.array([1, 2, 3, 4])

    def test_predict(self):
        y_pred, theta = closed_form_solution(self.X1, self.y1)
        y_pred2 = predict(self.X1, theta)
        self.assertTrue(np.allclose(y_pred, y_pred2))

        y_pred, theta = closed_form_solution(self.X2, self.y2)
        y_pred2 = predict(self.X2, theta)
        self.assertTrue(np.allclose(y_pred, y_pred2))

class TestComputeLoss(unittest.TestCase):
    def setUp(self):
        self.y_pred1 = np.array([1, 2, 3, 4])
        self.y1 = np.array([1, 2, 3, 4])

        self.y_pred2 = np.array([1.1, 2.2, 3.3, 4.4])
        self.y2 = np.array([1, 2, 3, 4])

    def test_compute_loss(self):
        loss = compute_loss(self.y_pred1, self.y1)
        self.assertEqual(loss, 0)

        loss = compute_loss(self.y_pred2, self.y2)
        self.assertAlmostEqual(loss, 0.075)

def run_closed_form_test():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestClosedFormSolution)
    result = unittest.TextTestRunner().run(suite)

    score = (result.testsRun - (len(result.failures) + len(result.errors))) / result.testsRun * 4
    print(f"Final Score of Closed Form Solution: {score}/4")

def run_predict_test():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPredict)
    result = unittest.TextTestRunner().run(suite)

    score = (result.testsRun - (len(result.failures) + len(result.errors))) / result.testsRun * 4
    print(f"Final Score of Predict: {score}/4")

def run_compute_loss_test():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestComputeLoss)
    result = unittest.TextTestRunner().run(suite)

    score = (result.testsRun - (len(result.failures) + len(result.errors))) / result.testsRun * 2
    print(f"Final Score of Compute Loss: {score}/2")

if __name__ == "__main__":
    unittest.main()
    