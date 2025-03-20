import numpy as np
from scipy.optimize import linear_sum_assignment

def clustering_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy after using the Hungarian algorithm to find the
    best matching between true and predicted labels.
    
    Args:
        y_true: true labels, numpy.array of shape (n_samples,)
        y_pred: predicted labels, numpy.array of shape (n_samples,)
        
    Returns:
        accuracy: float, clustering accuracy
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    assert y_pred.size == y_true.size, "Size of y_true and y_pred must be equal"
    
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    
    # Count the intersection between y_true and y_pred
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    # Use Hungarian algorithm to find the best matching
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    
    # Calculate accuracy
    count = sum([w[row_ind[i], col_ind[i]] for i in range(len(row_ind))])
    
    return count / y_pred.size
