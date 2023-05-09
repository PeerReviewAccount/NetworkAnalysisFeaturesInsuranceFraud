# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:00:42 2023

@author: u0130626
"""
import numpy as np
from scipy import linalg
from scipy.stats import chi2


def kernel(X,Y):
    '''

    Parameters
    ----------
    X : FLOAT
        The predicted score of a positive observation.
    Y : FLOAT
        The predicted score of a negative observation.

    Returns
    -------
    FLOAT
        The value of the kernel function for the DeLong DeLong Clarke-Pearson test.

    '''
    
    if X > Y:
        return 1
    elif X==Y:
        return 1/2
    else:
        return 0

def DL_DL_CP(D, theta, L):
    
    X = D[D["label"]==1] #All positive observations
    Y = D[D["label"]==0] #All negative observations
    
    n = len(Y)
    m = len(X)
    r = len(theta)
    
    V10 = np.zeros( (m,r) )
    
    for c in range(r):
        for i in range(m):
            V_sum = 0
            
            for j in range(n):
                V_sum += kernel(X.iloc[i,r], Y.iloc[j, r])
                
            V10[i,r] = V_sum/n
                
            
    V01 = np.zeros( (n,r) )
    
    for c in range(r):
        for j in range(n):
            V_sum = 0
            
            for i in range(m):
                V_sum += kernel(X.iloc[i,r], Y.iloc[j, r])
                
            V01[j,r] = V_sum/m
            
    S10 = (1/(m-1)) * ( np.transpose(V10)@V10 - m * (np.transpose(theta)@theta) )
    S01 = (1/(n-1)) * ( np.transpose(V01)@V01 - n * (np.transpose(theta)@theta) )

    S = S10/m + S01/n                
    
    variance = L @ S @ np.transpose(L)
    
    K = linalg.inv(variance)
    
    chi_stat = theta @ np.transpose(L) @ K @ L @ np.transpose(theta)

    df = int(np.linalg.matrix_rank(L))
    rv = chi2(df)
    
    pvalue = 1-rv.cdf(chi_stat)
    
    return(pvalue)