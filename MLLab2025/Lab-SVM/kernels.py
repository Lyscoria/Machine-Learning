import numpy as np


class Base_kernel():
    
    def __init__(self):
        pass
    
    def __call__(self, x1, x2):
        """
        Base kernel function.

        Arguments:
            x1 : np.ndarray, shape (n1, d) - First input data array.
            x2 : np.ndarray, shape (n2, d) - Second input data array.
            
        Returns:
            y : np.ndarray, shape (n1, n2), where y[i, j] = kernel(x1[i], x2[j]).
        """
        pass


class Linear_kernel(Base_kernel):
    
    def __init__(self):
        super().__init__()
    
    def __call__(self, x1, x2):
        # TODO: Implement the linear kernel function
        y = x1 @ x2.T
        return y
    
    
class Polynomial_kernel(Base_kernel):
        
    def __init__(self, degree, c):
        super().__init__()
        self.degree = degree
        self.c = c
        
    def __call__(self, x1, x2):
        # TODO: Implement the polynomial kernel function
        y = (x1 @ x2.T + self.c) ** self.degree
        return y

class RBF_kernel(Base_kernel):
    
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma 
        
        
    def __call__(self, x1, x2):
        # TODO: Implement the RBF kernel function
        if x1.ndim == 1:
            x1 = x1.reshape(1, -1)
        if x2.ndim == 1:
            x2 = x2.reshape(1, -1)
        x1_sq = np.sum(x1 ** 2, axis=1)[:, None]
        x2_sq = np.sum(x2 ** 2, axis=1)[None, :]
        y = np.exp(-(x1_sq + x2_sq - 2 * x1 @ x2.T) / (2 * self.sigma ** 2))
        return y