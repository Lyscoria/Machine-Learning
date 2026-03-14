import numpy as np
import matplotlib.pyplot as plt


def visualize_2D_dataset(x, y):
    """
    Visualize a 2D dataset.

    Arguments:
        x : np.ndarray - Feature matrix, must have 2 features (shape: (N, 2)).
        y : np.ndarray - Labels vector, where each label is either 1 (positive) or -1 (negative).

    Notes:
        This function requires x to be 2-dimensional for visualization. If x has higher dimensions,
        consider using PCA or another dimensionality reduction technique to reduce x to 2 dimensions.
    """
    
    assert x.shape[1] == 2, "x must be 2D for visualization."

    # Separate positive and negative samples for visualization
    x_pos = x[y == 1]
    x_neg = x[y == -1]
    
    plt.scatter(x_pos[:, 0], x_pos[:, 1], c='red', label='Positive')
    plt.scatter(x_neg[:, 0], x_neg[:, 1], c='blue', label='Negative')
    plt.title('Zipper-like Dataset')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def visualize_2D_border(svm, x1_interval, x2_interval, x_train=None, y_train=None):
    """
    Visualize the decision boundary of a trained SVM model in 2D space.

    Arguments:
        svm : object - Trained SVM model that has a predict function.
        x1_interval : tuple - Range (min, max) for the x1 axis.
        x2_interval : tuple - Range (min, max) for the x2 axis.
        x_train : np.ndarray, optional - Training feature matrix, shape (N, 2), used to display training points.
        y_train : np.ndarray, optional - Training labels vector, where each label is either 1 (positive) or -1 (negative).

    Notes:
        If x_train and y_train are provided, the function will also plot the training points.
    """
    
    # Generate a grid of points within the specified intervals
    x1_grid, x2_grid = np.meshgrid(np.linspace(x1_interval[0], x1_interval[1], 100),
                                   np.linspace(x2_interval[0], x2_interval[1], 100))
    x_grid = np.concatenate([x1_grid.reshape(-1, 1), x2_grid.reshape(-1, 1)], axis=1)
    
    # Predict labels for each point in the grid
    pred_y = svm.predict(x_grid)[1]
    
    # Plot decision boundary
    plt.contourf(x1_grid, x2_grid, pred_y.reshape(x1_grid.shape), cmap=plt.cm.coolwarm, alpha=0.8)
    plt.contour(x1_grid, x2_grid, pred_y.reshape(x1_grid.shape), colors='k', linewidths=0.5)
    plt.title('Decision Boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    
    # If training data is provided, plot it
    if x_train is not None and y_train is not None:
        x_pos = x_train[y_train == 1]
        x_neg = x_train[y_train == -1]
        
        plt.scatter(x_pos[:, 0], x_pos[:, 1], c='red', label='Positive')
        plt.scatter(x_neg[:, 0], x_neg[:, 1], c='blue', label='Negative')
        plt.legend()
    
    plt.show()