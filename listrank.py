"""
Implements our list rank algorithm
"""
from scipy.stats import norm
from math import log, pi
import numpy as np

from generate import generate

# TODO - tests for theta

# TODO speed this up with numpy

# TODO - compare both methods (need metrics)

def ll(theta, results):
    """ Get log likelihood of results given theta"""
    return sum([log(norm.cdf(theta[x]-theta[y])) for (x,y) in results])
def gradll(theta, resw, resl, scale=1):
    """ Gradient of log likelihood of reslts at theta"""
    g = np.zeros(len(theta))
    gett = lambda x : theta[x]
    tw = gett(resw)
    tl = gett(resl)
    a = scale
    # We use the vectorized versino of norm.pdf and norm.cdf to speed up this bottleneck
    risk = norm.pdf(a*(tw-tl)) / norm.cdf(a*(tw-tl))
    # Optimization candidate
    for i in range(len(theta)):
        x = resw[i]
        y = resl[i]
        g[x] += a*risk[i]
        g[y] -= a*risk[i]
    g -= theta # ridge regressions. Constant comes from standard normal prior
    da = sum((tw-tl)*risk) # Update scale constant
    return g, da
def order(theta):
    """ Return the theta value """
    o = list(range(len(theta)))
    o.sort(key = lambda x : theta[x], reverse=True)
    return o

### 
def train_basic(n=30):
    """
    Basic model witout scale parameter
    """
    results = generate(n, compf="normal")
    theta = np.full(n, 0.5) # initial guess
    alpha = 0.01 # step size
    gsize = 100 # big number
    it = 0
    # Pre-parse the results into numpy for faster performance
    res = results[1]
    resw = np.array([a for (a,_) in res], dtype=np.uint32)
    resl = np.array([b for (_,b) in res], dtype=np.uint32)
    while gsize > 0.01 and it < 10000:
        delta, _ = gradll(theta, resw, resl)
        theta += alpha*delta
        gsize = np.linalg.norm(delta)
        it += 1
    print(order(theta))
    print(order(results[0]))

def train_scale(n=30, scale=1, c=None):
    """
    Introduce scale into the approximations
    """
    results = generate(n, compf="normal", scale=scale, c=c)
    theta = np.full(n, 0.0) # initial guess
    a = 5
    alpha = 0.01 # step size
    gsize = 100 # big number
    it = 0
    # Pre-parse the results into numpy for faster performance
    res = results[1]
    resw = np.array([a for (a,_) in res], dtype=np.uint32)
    resl = np.array([b for (_,b) in res], dtype=np.uint32)
    imax = 10000
    thresh = 1e-12
    while gsize > thresh and it < imax:
        delta, da = gradll(theta, resw, resl, a)
        theta += alpha*delta
        a += alpha*da
        gsize = np.linalg.norm(delta) + da**2
        it += 1
    if it >= imax:
        print("Max out iterations")
    print(order(theta))
    print(order(results[0]))
    print(a)

if __name__ == "__main__":
    train_scale(5, 5, c=10)
