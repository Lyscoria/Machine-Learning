from kernels import *
from SVM import *
from SSMO import *
import numpy as np

def train(x_train, y_train):
    """
    Train an SVM model using the SSMO optimizer and an RBF kernel.

    Arguments:
        x_train : np.ndarray, shape (N, D) - Training data, where N is the number of samples and D is the number of features.
        y_train : np.ndarray, shape (N,) - Training labels, where each label is {-1, 1}.
    
    Returns:
        svm : Trained SVM model.
    """
    # TODO: Initialize an SVM model and train it with SSMO optimizer with your hyper-parameters.
    kernel_fn = RBF_kernel(sigma=1.0)
    svm = SVM(kernel_fn)
    optimizer = SSMO_optimizer(C=10.0)
    optimizer.fit(svm, x_train, y_train, max_passes=1000, verbose=False)

    return svm