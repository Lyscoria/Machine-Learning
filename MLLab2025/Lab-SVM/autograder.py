import numpy as np
import unittest
from kernels import *
from SVM import *
from SSMO import *
from dataset_generator import *
from train import *

class TestKernels(unittest.TestCase):
    def setUp(self):
        # Test data
        self.x1 = np.array([[1, 2], [3, 4]])
        self.x2 = np.array([[5, 6], [7, 8]])
    
    def test_linear_kernel(self):
        # Expected output
        expected_result = np.array([[17, 23],
                                    [39, 53]])
        
        # Test Linear Kernel
        linear_kernel = Linear_kernel()
        result = linear_kernel(self.x1, self.x2)
        
        # Assertion with error message
        np.testing.assert_array_almost_equal(
            result, expected_result, decimal=5,
            err_msg="Linear kernel output does not match the expected values."
        )

    def test_polynomial_kernel(self):
        # Parameters for Polynomial kernel
        degree = 2
        c = 1
        
        # Expected output
        expected_result = np.array([[324, 576],
                                    [1600, 2916]])
        
        # Test Polynomial Kernel
        poly_kernel = Polynomial_kernel(degree=degree, c=c)
        result = poly_kernel(self.x1, self.x2)
        
        # Assertion with error message
        np.testing.assert_array_almost_equal(
            result, expected_result, decimal=5,
            err_msg="Polynomial kernel output does not match expected values with degree=2 and c=1."
        )

    def test_rbf_kernel(self):
        # Parameter for RBF kernel
        sigma = 3.0
        
        # Expected output
        expected_result = np.array(
            [[0.16901332, 0.01831564],
             [0.64118039, 0.16901332]]
        )
        
        # Test RBF Kernel
        rbf_kernel = RBF_kernel(sigma=sigma)
        result = rbf_kernel(self.x1, self.x2)
        
        # Assertion with error message
        np.testing.assert_array_almost_equal(
            result, expected_result, decimal=5,
            err_msg="RBF kernel output does not match expected values with sigma=3.0."
        )

class TestSVM(unittest.TestCase):
    def setUp(self):
        # Initialize the SVM object with the RBF kernel
        self.svm = SVM(kernel_fn=RBF_kernel(sigma=3.0))
        
        # Mock data for SVM's parameters
        self.svm.b = 0.5  # Bias term
        self.svm.alpha = np.array([0.3, 0.7])  # Dual variables (example values)
        self.svm.support_labels = np.array([1, -1])  # Support labels in {-1, 1}
        self.svm.support_vectors = np.array([[1, 2], [3, 4]])  # Support vectors
        
        self.x_test = np.array([[2, 3], [1, -1]])

    def test_predict(self):
        # Expected output
        expected_scores = np.array([0.14206427, 0.54219325])
        expected_pred = np.array([1., 1.])

        # Run predict
        scores, pred = self.svm.predict(self.x_test)
        
        # Assertions with error messages
        np.testing.assert_array_almost_equal(
            scores, expected_scores, decimal=5,
            err_msg="SVM predict scores do not match expected values."
        )
        np.testing.assert_array_equal(
            pred, expected_pred,
            err_msg="SVM predict labels do not match expected values."
        )


class TestSSMOOptimizer(unittest.TestCase):
    
    def setUp(self):
        # Initialize a dummy SVM with a linear kernel
        np.random.seed(42)
        self.svm = SVM(kernel_fn=Linear_kernel())
        
        # Mock data for SVM's parameters
        self.svm.b = 0.5  # Bias term
        self.svm.alpha = np.array([0.3, 0.7])  # Dual variables (example values)
        self.svm.support_labels = np.array([1, -1])  # Support labels in {-1, 1}
        self.svm.support_vectors = np.array([[1, 2], [3, 4]])  # Support vectors
        
        # Initialize SSMO optimizer with penalty parameter C and KKT threshold
        self.optimizer = SSMO_optimizer(C=0.3, kkt_thr=1e-3)
        self.optimizer.SVM = self.svm  # Link the SVM to the optimizer

        # Set up mock values for the test cases
        self.optimizer.alpha = np.array([0.5, 0.2])
        self.optimizer.b = 0.1

        # Training data and labels
        self.x_train = np.array([[1, 2], [3, 4]])
        self.y_train = np.array([1, -1])

        
    def test_compute_L_H(self):
        # Test case 1: When y[i] == y[j]
        L, H = self.optimizer.compute_L_H(is_yi_equals_yj=True, i=0, j=1)
        self.assertAlmostEqual(L, 0.4, msg="Failed L boundary when y[i] == y[j]")
        self.assertAlmostEqual(H, 0.3, msg="Failed H boundary when y[i] == y[j]")

        # Test case 2: When y[i] != y[j]
        L, H = self.optimizer.compute_L_H(is_yi_equals_yj=False, i=0, j=1)
        self.assertAlmostEqual(L, 0, msg="Failed L boundary when y[i] != y[j]")
        self.assertAlmostEqual(H, 0, msg="Failed H boundary when y[i] != y[j]")

    def test_compute_new_aj_when_eta_positive(self):
        # Set a positive eta
        eta = 0.5
        L, H = 0, 1.0
        
        expected_aj_new = 0

        # Run compute_new_aj
        aj_new = self.optimizer.compute_new_aj(
            x_train=self.x_train,
            y_train=self.y_train,
            i=0, j=1, eta=eta, L=L, H=H
        )

        # Assertion with message
        self.assertAlmostEqual(aj_new, expected_aj_new, places=5,
                               msg="compute_new_aj did not return expected value when eta > 0")

    def test_compute_new_ai(self):
        # Assume aj_new was computed as follows
        aj_new = 0.4

        # Expected ai_new based on the formula for self.alpha[i]
        expected_ai_new = 0.7

        # Run compute_new_ai
        ai_new = self.optimizer.compute_new_ai(y_train=self.y_train, i=0, j=1, aj_new=aj_new)

        # Assertion with message
        self.assertAlmostEqual(ai_new, expected_ai_new, places=5,
                               msg="compute_new_ai did not return the expected value")




class Test_train(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        n_pos = 100
        n_neg = 100
        self.x_train, self.y_train = zipper_2D_dataset(n_pos, n_neg, scope=4.0)
        self.x_test, self.y_test = zipper_2D_dataset(50, 50, scope=5.5)  # Test data slightly out-of-distribution

    def test_train(self):
        # Train SVM model on training data
        svm = train(self.x_train, self.y_train)
        
        # Calculate training accuracy
        y_pred_train = svm.predict(self.x_train)[1]
        acc_train = np.sum(y_pred_train == self.y_train) / len(self.y_train)
        
        # Calculate test accuracy
        y_pred_test = svm.predict(self.x_test)[1]
        acc_test = np.sum(y_pred_test == self.y_test) / len(self.y_test)
        
        # Assert that the training and test accuracies meet the thresholds
        self.assertGreater(acc_train, 0.99, "Training accuracy is below the required threshold of 97%")
        self.assertGreater(acc_test, 0.97, "Test accuracy is below the required threshold of 95%")

        

# Run the SVM tests
def run_svm_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSVM)
    result = unittest.TextTestRunner().run(suite)
    
    # Calculate the score based on tests passed
    score = (result.testsRun - (len(result.failures) + len(result.errors))) * 3
    print(f"Final Score of SVM predict: {score}/3")


# Run the Kernel tests
def run_kernel_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKernels)
    result = unittest.TextTestRunner().run(suite)
    
    # Calculate the score based on tests passed
    score = (result.testsRun - (len(result.failures) + len(result.errors)))
    print(f"Final Score of Kernel Functions: {score}/3")


# Run the SVM tests
def run_SSMO_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSSMOOptimizer)
    result = unittest.TextTestRunner().run(suite)
    
    # Calculate the score based on tests passed
    score = (result.testsRun - (len(result.failures) + len(result.errors))) * 8 / 3 
    print(f"Final Score of SSMO predict: {score}/8")
    
def run_train_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(Test_train)
    result = unittest.TextTestRunner().run(suite)
    
    # Calculate the score based on tests passed
    score = (result.testsRun - (len(result.failures) + len(result.errors))) * 6
    print(f"Final Score of train: {score}/6")
    