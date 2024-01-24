import numpy as np

def itr(acc_matrix, M, t):
    """
    Information Transfer Rate (ITR) calculation.

    Input:
        - acc_matrix: Accuracy matrix
        - M: Number of characters
        - t: Total signal length (visual cue + desired signal length)

    Output:
        - ITR: Information Transfer Rate
    """
    size_mat = np.shape(acc_matrix)
    itr_matrix = np.zeros(size_mat)
    for i in range(size_mat[0]):
        for j in range(size_mat[1]):
            p = acc_matrix[i, j]
            if p < 1/M:
                itr_matrix[i, j] = 0
            elif p == 1:
                itr_matrix[i, j] = np.log2(M) * (60/t)
            else:
                itr_matrix[i, j] = (np.log2(M) + p*np.log2(p) + (1-p)*np.log2((1-p)/(M-1))) * (60/t)
    return itr_matrix
