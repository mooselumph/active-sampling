import numpy as np
import numpy.random as random

def get_data(size, sigma=0.1):

    t = np.linspace(-1,1,size)
    X,Y = np.meshgrid(t,t)

    d = (X - Y)**2

    cov = np.exp(-d / sigma**2) 
    mu = np.zeros(size)

    y = random.multivariate_normal(mu,cov)

    return y
    