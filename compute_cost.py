import numpy as np


class ComputeCost:
    def __init__(self, ):
        pass

    def compute_cost(self, X, y, theta):
        m = len(y)  # number of training samples
        J = 0
        X = X.transpose()
        y = y.reshape(m, 1)
        hypothesis = X.dot(theta)
        J = 1 / (2 * m) * sum(np.square(hypothesis - y))
        return J
