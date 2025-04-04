"""
Entropy/Anisotropy/Alpha (H/A/alpha) decomposition of a 3x3 polarimetric coherency matrix.

S. R. Cloude and E. Pottier
A review of target decomposition theorems in radar polarimetry
IEEE Transactions on Geoscience and Remote Sensing, 1996
DOI: 10.1109/36.485127
"""

import numpy as np


def h_a_alpha_decomposition(coherency_matrix):
    """
    Entropy/Anisotropy/Alpha (H/A/alpha) decomposition of a 3x3 polarimetric coherency matrix.
    Parameters:
        coherency_matrix - polarimetric coherency matrices as (*B, 3, 3) numpy array, where B are the batch dimensions
    Returns:
        A tuple of entropy, anisotropy, mean alpha angle, and dominant alpha angle.
        In case an individual 3x3 matrix is decomposed, the tuple contains scalar values.
        In the batched case, the tuple contains arrays that have the shape of the batch dimensions.
    """
    t_eigval, t_eigvec = np.linalg.eigh(coherency_matrix)
    pseudo_probs = t_eigval / np.sum(t_eigval, axis=-1, keepdims=True)
    # alpha angles
    alpha_i = np.arccos(np.abs(t_eigvec[..., 0, :]))  # arccos of abs of first element of each eigenvector
    alpha_mean = np.sum(pseudo_probs * alpha_i, axis=-1)  # average alpha angle (weighted by eigenvalues)
    alpha_dominant = alpha_i[..., 2]  # alpha angle corresponding to the largest eigenvalue (at index 2)
    # entropy
    log_pseudo_probs = np.log(pseudo_probs, out=np.zeros_like(pseudo_probs), where=pseudo_probs > 1e-5)
    entropy = -np.sum(pseudo_probs * log_pseudo_probs, axis=-1) / np.log(3)  # log base 3
    # anisotropy
    t_eigval_2 = t_eigval[..., 1]  # second (intermediate) eigenvalue has index 1
    t_eigval_3 = t_eigval[..., 0]  # third (smallest) has index 0 (ascending sorting)
    t_eigval_sum = t_eigval_2 + t_eigval_3
    anisotropy = np.divide(
        t_eigval_2 - t_eigval_3, t_eigval_sum, out=np.zeros_like(t_eigval_sum), where=t_eigval_sum != 0
    )
    return entropy, anisotropy, alpha_mean, alpha_dominant
