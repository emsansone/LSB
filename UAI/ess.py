import numpy as np

# Computing the effective sample size
# Input:
# x - numpy array of size n x d (autocorrelation scores)
# Output:
# mean_ess - average effecive sample size
# std_ess - std. dev. effective sample size
def effective_sample_size(x):
    n, d = x.shape
    ess = np.zeros((n,))
    for i in range(n):
        for j in range(1, d):
            if x[i, j] > 0.05: # Threshold selected based on code of Zanella's paper 2020
                ess[i] = ess[i] + (x[i, j] * (1. - j / float(d)))
    ess = float(d) / (1. + 2. * ess)
    mean_ess = np.mean(ess)
    std_ess = np.std(ess)
    return mean_ess, std_ess
