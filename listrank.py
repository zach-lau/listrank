"""
Implements our list rank algorithm
"""
from scipy.stats import norm
from math import log, pi
import numpy as np

from generate import generate

# TODO - tests for theta

# TODO speed this up with numpy

def ll(theta, results):
    """ Get log likelihood of results given theta"""
    return sum([log(norm.cdf(theta[x]-theta[y])) for (x,y) in results])
def gradll(theta, resw, resl):
    """ Gradient of log likelihood of reslts at theta"""
    g = np.zeros(len(theta))
    gett = lambda x : theta[x]
    tw = gett(resw)
    tl = gett(resl)
    # We use the vectorized versino of norm.pdf and norm.cdf to speed up this bottleneck
    risk = norm.pdf(tw-tl) / norm.cdf(tw-tl)
    # Optimization candidate
    for i in range(len(theta)):
        x = resw[i]
        y = resl[i]
        g[x] += risk[i]
        g[y] -= risk[i]
    g -= theta # ridge regressions. Constant comes from standard normal prior
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
    # Pre-parse the results into numpy for faster performance
    res = results[1]
    resw = np.array([a for (a,_) in res], dtype=np.uint32)
    resl = np.array([b for (_,b) in res], dtype=np.uint32)
    while gsize > 0.01 and it < 10000:
        delta = gradll(theta, resw, resl)
        theta += alpha*delta
        gsize = np.linalg.norm(delta)
        it += 1
    print(order(theta))
    print(order(results[0]))
