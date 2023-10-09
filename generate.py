"""
This file should genrate dummy data which we can use to test our inference algorithm
"""
from math import log, ceil
from random import sample
# TODO merge dependencies here
from numpy.random import normal, uniform
from scipy.stats import norm

# Tests to do 
# - test with identity compfunc and fixed theta
# - test that our probability of getting one sample of each is calculated correctly
def generate(n=10, compf = "basic"):
    """
    Return a list of randomly generated comparisons. The first element is deemed better up to fuzzing than teh second
    """
    func_dict = {
        "basic" : lambda x,y : 1 if x > y else 0,
        "normal" : lambda x,y : norm.cdf(x-y)
    }
    cfunc = func_dict[compf]
    alpha = 0.01
    c = ceil( log(alpha) / log(1-2/n) ) # number of comparisons needed for 1-alpha% chance each element shows up in at least
    # one comparison
    theta = normal(0, 1, n) # true value of each parameter
    results = []
    # TODO vectorize this
    for _ in range(c):
        a,b = sample(range(n), 2)
        s,t = a,b
        if uniform() > cfunc(theta[a],theta[b]):
            s,t = b,a
        results.append((s,t))
    return theta, results

def test_simple():
    print("Test - simple")
    n = 5
    t, w = generate(n)
    o = list(range(n))
    o.sort(key=lambda x : t[x], reverse=True)
    print(o)
    print(t)
    print(w)

def test_normal():
    """ This test uses a normal cdf comparison function"""
    print("Test - normal")
    n = 5
    t, w = generate(n, compf="normal")
    o = list(range(n))
    o.sort(key=lambda x : t[x], reverse=True)
    print(o)
    print(t)
    print(w)

if __name__ == "__main__":
    print(generate(100, compf="normal")[1])
