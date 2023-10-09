"""
Implements our list rank algorithm
"""
from scipy.stats import norm
from math import log
import numpy as np

from generate import generate

# TODO - tests for theta

# TODO speed this up with numpy

def ll(theta, results):
    """ Get log likelihood of results given theta"""
    return sum([log(norm.cdf(theta[x]-theta[y])) for (x,y) in results])
def gradll(theta, results):
    """ Gradient of log likelihood of reslts at theta"""
    g = np.zeros(len(theta))
    # Optimization candidate
    for x,y in results:
        diff = norm.pdf(theta[x]-theta[y]) / norm.cdf(theta[x]-theta[y])
        g[x] += diff
        g[y] -= diff
    return g
def order(theta):
    """ Return the theta value """
    o = list(range(len(theta)))
    o.sort(key = lambda x : theta[x], reverse=True)
    return o

if __name__ == "__main__":
    # we store a list of results in a list
    n = 30
    results = generate(n, compf="normal")
    theta = np.full(n, 0.5) # initial guess
    alpha = 0.01 # step size
    gsize = 100 # big number\
    it = 0
    while gsize > 0.1 and it < 10000:
        delta = gradll(theta, results[1])
        theta += alpha*delta
        gsize = np.linalg.norm(delta)
        it += 1
    print(order(theta))
    print(order(results[0]))
