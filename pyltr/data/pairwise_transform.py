import numpy as np
import itertools


def pairwise_transform(x, y):
    
    '''
    Performs the pairwise transformation of the input data as described in 
    Herbrich, R., Graepel, T., & Obermayer, K. (1999). Support vector learning for ordinal regression.
    (x_i, y_i) = (x_i - x_j, sign(y_i - y_j))
    
    WARNING: may be computationally expensive due to slow itertools.combinations
    and growing size of unique combinations
    
    Arguments:
        x: input data as list, pandas dataframe or numpy array of shape (num_samples, num_features)
        y: labels as list, pandas dataframe or numpy array of shape (num_samples)
    Return:
        xpair: input data after pairwise transform as numpy array of shape (num_pairs, num_features)
        ypair: labels after pairwise transform (values either -1 or 1]) as numpy array of shape (num_pairs)
    '''
    
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    comb_iter = itertools.combinations(range(x.shape[0]), 2)
    comb_vals = [row for row in comb_iter]
     
    xpair, ypair = list(), list()
    balance = False
    
    for i,j in comb_vals:
        if y[i] == y[j]:
            continue
        else:
            xpair.append(x[i] - x[j])
            ypair.append(np.sign(y[i] - y[j]))
        if balance == True:
            balance = False
            continue
        else:
            balance = True
            xpair[-1] = np.negative(xpair[-1])
            ypair[-1] = np.negative(ypair[-1])
    
    return np.asarray(xpair), np.asarray(ypair)