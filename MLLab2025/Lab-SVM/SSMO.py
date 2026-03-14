import numpy as np
from tqdm import tqdm

class SSMO_optimizer():
    
    """
    Simplified Sequential Minimal Optimization (SSMO) algorithm for training SVM.
    """
    
    def __init__(
        self,
        C: float = 1.,
        kkt_thr: float = 1e-3,
    ) -> None:

        """
        Arguments:
            C : Penalty parameter for SVM.
            kkt_thr : Threshold for KKT conditions, typically set to 1e-3.
        """
        # Initialize parameters
        self.C = float(C)
        self.kkt_thr = kkt_thr
        
        self.b = None
        self.alpha = None
        self.support_vectors = None
        self.support_labels = None


    def fit(
        self,
        SVM,
        x_train: np.ndarray,
        y_train: np.ndarray,
        max_passes: int = 1000,
        verbose: bool = False,
    ) -> None:

        """
        Train an SVM classifier.

        Arguments:
            SVM : The SVM classifier to be trained. The training process will update the SVM's attributes:
                  self.b, self.alpha, self.support_vectors, and self.support_labels.
            x_train : (N, D) data matrix, where each row is a sample.
            y_train : Labels vector, y must be {-1,1}.
            max_passes : Maximum number of passes through the training data.
            verbose : If True, prints the loss during training.
            
        Return:
            None
            Note that the training process will overwrite the SVM's attributes, so the trained SVM can be accessed 
            through SSMO_optimizer.SVM directly after training.
        """
        np.random.seed(0)

        # Initialize variables
        N, _ = x_train.shape
        self.SVM = SVM
        
        # Initialize SVM attributes to ensure they satisfy the constraints of the dual form of SVM.
        self.b = 0
        self.alpha = np.zeros(N)
        self.support_vectors = x_train
        self.support_labels = y_train
        self.update_SVM()

        passes = 0

        print("Start training using SMO")

        bar = tqdm()
        
        while passes < max_passes:
            num_changed_alphas = 0 # Counts the number of alpha variables updated in this pass; when num_changed_alphas == 0, the training process should terminate.
            for i in range(N):
                # TODO: implement SSMO algorithm for each pass (One pass means one iteration over the training data)
                # Hint: 1. First select the two alpha variables to be updated, i.e., alpha[i] and alpha[j], based on the heuristic method in SVM_lab.ipynb.
                #       2. Then, use the constraint in the dual form of SVM to express alpha[i] in terms of alpha[j] and optimize the polynomial function of alpha[j].
                #       3. Finally, update alpha[i] and alpha[j] based on the optimal solution of alpha[j].
                # Note: 1. Remember to call self.update_SVM() after updating self.b and self.alpha (self.support_vectors and self.support_labels).
                #       2. If eta is 0., skip this pair of alpha variables. 
                if self.judge_violoate_KKT(x_train[i], y_train[i], self.alpha[i]):

                    j = i
                    while j == i:
                        j = np.random.randint(0, N)
                    L, H = self.compute_L_H(y_train[i] == y_train[j], i, j)
                    if L == H:
                        continue
                    eta = self.compute_eta(x_train[i], x_train[j])
                    if eta <= 0:
                        continue

                    aj_new = self.compute_new_aj(x_train, y_train, i, j, eta, L, H)
                    if abs(aj_new - self.alpha[j]) < 1e-5:
                        continue

                    ai_new = self.compute_new_ai(y_train, i, j, aj_new)
                    b_new = self.compute_new_b(x_train, y_train, i, j, ai_new, aj_new)

                    self.alpha[i] = ai_new
                    self.alpha[j] = aj_new
                    self.b = b_new

                    self.update_SVM()
                    num_changed_alphas += 1
            
            if verbose: 
                loss = self.eval_loss(x_train, y_train) # Report the loss after each pass
                bar.set_description(f"loss: {loss:.4f}")
            
            if num_changed_alphas == 0:
                break
            else:
                passes += 1
                bar.update(1)

        # Keep only the support vectors
        self.support_labels = self.support_labels[self.alpha > 0]
        self.support_vectors = self.support_vectors[self.alpha > 0]
        self.alpha = self.alpha[self.alpha > 0]
        self.update_SVM()
        print("Training finished")
        return 

    def judge_violoate_KKT(
        self,
        xi,
        yi,
        alpha_i
    ) -> bool:
        """
        Determine whether the i-th sample violates the KKT conditions.
        
        Arguments:
            xi : (D,), the i-th sample.
            yi : (1,), the label of the i-th sample, must be {-1, 1}.
            alpha_i : The dual variable for the i-th sample.
            
        Return:
            True if the i-th sample violates KKT conditions, otherwise False.
            
        Note:
            The following conditions are derived directly from the KKT conditions of the dual form of SVM. This is one heuristic way to check whether the i-th sample is 
            suitable for optimization. kkt_thr is a numerical threshold, usually set to 1e-3.
        """
        
        if alpha_i == 0:
            return yi * self.predict_score(xi) < 1 - self.kkt_thr
        elif alpha_i == self.C:
            return yi * self.predict_score(xi) > 1 + self.kkt_thr
        else:
            return abs(yi * self.predict_score(xi) - 1) > self.kkt_thr
    
    
    def compute_E(
        self,
        xi,
        yi
    ):
        """
        Compute the prediction error for the i-th sample.
        
        Arguments:
            xi : (D,), the i-th sample.
            yi : (1,), the label of the i-th sample, must be {-1, 1}.
            
        Return:
            The prediction error for the i-th sample.
        """
        return self.predict_score(xi) - yi
    
    
    def compute_L_H(
        self,
        is_yi_equals_yj,
        i,
        j,
    ):
        """
        Compute the lower and upper bounds of alpha[j] based on the dual form constraint of SVM, i.e., 0 <= alpha[i] <= C and sum_k alpha[k] * y[k] = 0.
        
        Arguments:
            is_yi_equals_yj : True if y[i] == y[j], otherwise False.
            i : Index of the i-th sample.
            j : Index of the j-th sample.
            
        Return:
            L : The lower bound of **alpha[j]**.
            H : The upper bound of **alpha[j]**.
        """
        # TODO: implement compute_L_H. You may use only self.C and self.alpha in this function.
        if is_yi_equals_yj:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])
        else:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])

        return L, H
        
    def compute_eta(self, xi, xj):
        '''
        Compute eta, the second derivative of the dual form objective function for SVM (the equivalent minimization problem).
        
        Arguments:
            xi : (D,), the i-th sample.
            xj : (D,), the j-th sample.
        
        Return:
            eta : The second derivative of the dual form objective function.
        '''
        
        eta = (
            - 2 * self.kernel_fn(xi, xj) 
            + self.kernel_fn(xi, xi)
            + self.kernel_fn(xj, xj)
        )
        eta = eta.squeeze()
        
        return eta
    
    
    def compute_new_aj(self, x_train, y_train, i, j, eta, L, H):
        '''
        Compute the updated value of alpha[j].

        Arguments:
            x_train : (N, D), training data.
            y_train : (N,), labels for training data.
            i : Index of the i-th sample.
            j : Index of the j-th sample.
            eta : Second derivative of the dual form objective function.
            L : Lower bound of alpha[j].
            H : Upper bound of alpha[j].
            
        Return:
            aj_new : Updated value of alpha[j].
            
        Note:
            This is equivalent to solving the quadratic optimization problem for alpha[j] (alpha[i] is expressed in terms of alpha[j] based on the dual form constraint).
        '''
        
        if eta > 0:
            # TODO: implement compute_new_aj when eta > 0. 
            # Hint: By setting the first derivative of the dual form objective function to 0, 
            #       one can derive that the unbounded optimal solution of alpha_j_opt is alpha_j_opt = alpha_j_old + y_j * (E_i - E_j) / eta.
            Ei = self.compute_E(x_train[i], y_train[i])
            Ej = self.compute_E(x_train[j], y_train[j])
            aj_new = self.alpha[j] + y_train[j] * (Ei - Ej) / eta

            if aj_new > H:
                aj_new = H
            if aj_new < L:
                aj_new = L
            
            if hasattr(aj_new, 'item'):
                return aj_new.item()
            return aj_new
        else:
            # Evaluate objective function at a1 = L and a1 = H
            
            Ei = self.compute_E(x_train[i], y_train[i])
            Ej = self.compute_E(x_train[j], y_train[j])
            kii = self.kernel_fn(x_train[i], x_train[i]).squeeze()
            kij = self.kernel_fn(x_train[i], x_train[j]).squeeze()
            kjj = self.kernel_fn(x_train[j], x_train[j]).squeeze()
            s = y_train[i] * y_train[j]
            
            f1 = y_train[i] * (Ei + self.b) - self.alpha[i] * kii - s * self.alpha[j] * kij
            f2 = y_train[j] * (Ej + self.b) - s * self.alpha[i] * kij - self.alpha[j] * kjj
            L1 = self.alpha[i] + s * (self.alpha[j] - L)
            H1 = self.alpha[i] + s * (self.alpha[j] - H)
            
            L_obj = L1 * f1 + L * f2 + 0.5 * L1 ** 2 * kii + 0.5 * L ** 2 * kjj + s * L * L1 * kij
            H_obj = H1 * f1 + H * f2 + 0.5 * H1 ** 2 * kii + 0.5 * H ** 2 * kjj + s * H * H1 * kij
            
            if L_obj < H_obj - 1e-3:
                aj_new = L
            elif L_obj > H_obj + 1e-3:
                aj_new = H
            else:
                aj_new = self.alpha[j]
                
            return aj_new
        
    
    
    def compute_new_ai(self, y_train, i, j, aj_new):
        '''
        Compute the updated value of alpha[i].

        Arguments:
            y_train : (N,), training data labels.
            i : Index of the i-th sample.
            j : Index of the j-th sample.
            aj_new : Updated value of alpha[j].
            
        Return:
            ai_new : Updated value of alpha[i].
        '''
        
        
        # TODO: implement compute_new_ai. You may only use self.alpha additionally in this function.
        ai_new = self.alpha[i] + y_train[i] * y_train[j] * (self.alpha[j] - aj_new)
        if hasattr(ai_new, 'item'):
            return ai_new.item()
        return ai_new
    
    def compute_new_b(self, x_train, y_train, i, j, ai_new, aj_new):
        '''
        Compute the updated value of b based on the updated values of alpha[i] and alpha[j].
        
        Arguments:
            x_train : (N, D), training data.
            y_train : (N,), labels for training data.
            i : Index of the i-th sample.
            j : Index of the j-th sample.
            ai_new : Updated value of alpha[i].
            aj_new : Updated value of alpha[j].
        
        Return:
            b_new : Updated value of b.
        '''
        
        Ei = self.compute_E(x_train[i], y_train[i])
        Ej = self.compute_E(x_train[j], y_train[j])
        kii = self.kernel_fn(x_train[i], x_train[i]).squeeze()
        kjj = self.kernel_fn(x_train[j], x_train[j]).squeeze()
        kij = self.kernel_fn(x_train[i], x_train[j]).squeeze()
        
        
        bi = -(Ei + y_train[i] * (ai_new - self.alpha[i]) * kii + y_train[j] * (aj_new - self.alpha[j]) * kij) + self.b
        bj = -(Ej + y_train[i] * (ai_new - self.alpha[i]) * kij + y_train[j] * (aj_new - self.alpha[j]) * kjj) + self.b
        
        if 0 < ai_new < self.C:
            b_new = bi
        elif 0 < aj_new < self.C:
            b_new = bj
        else:
            b_new = (bi + bj) / 2
            
        return b_new
    
    
    def predict_score(self, x):
        return self.SVM.predict(x)[0]
    
    def kernel_fn(self, x1, x2):
        return self.SVM.kernel_fn(x1, x2)
    
    def update_SVM(self):
        # Update the four attributes of the SVM after each pass
        self.SVM.b = self.b
        self.SVM.alpha = self.alpha
        self.SVM.support_vectors = self.support_vectors
        self.SVM.support_labels = self.support_labels
        
    def eval_loss(self, x_train, y_train):
        # Evaluate the loss of the current SVM model with respect to the training data
        alpha_M = self.alpha[:, np.newaxis] * self.alpha[np.newaxis, :]
        y_train_M = y_train[:, np.newaxis] * y_train[np.newaxis, :]
        kernel_M = self.kernel_fn(x_train, x_train)
        
        loss = (0.5 * kernel_M * alpha_M * y_train_M).sum() - self.alpha.sum()
            
        return loss