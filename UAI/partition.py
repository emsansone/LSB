import itertools
import numpy as np

def log_partition(xs, model, evidence):
    result = 0
    for i in range(len(xs)):
        if i % 10 == 0:
            print('{}/{}'.format(i,len(xs)))

        # Computing denominator of score
        list_factors = []
        factors = model.factors
        evid_vars = set(evidence.keys())
        denominator = 1.
        for factor in factors:
            temp = set(factor.variables)
            evid_vars_factors = temp.intersection(evid_vars)
            if len(evid_vars_factors) != 0:
                phi = factor.copy()
                arg = [(v, xs[i][v]) for v in phi.variables]
                phi.reduce(arg)
                denominator *= phi.values
                list_factors.append(factor)

        # Storing values of xs
        xs_temp = {}
        for j in xs[i].keys():
            xs_temp[j] = xs[i][j]

        # Computing numerator of score
        combinations = list(itertools.product([0, 1], repeat=len(evidence)))
        evid_vars = list(evidence.keys())
        numerator = 0.
        for comb in combinations:
            for j in range(len(evid_vars)):
                xs_temp[evid_vars[j]] = comb[j]
            temp = 1.
            for factor in list_factors:
                phi = factor.copy()
                arg = [(v, xs_temp[v]) for v in phi.variables]
                phi.reduce(arg)
                temp *= phi.values
            numerator += temp
        result += (numerator / denominator)
    
    result = - np.log10(result / len(xs))
    return result
