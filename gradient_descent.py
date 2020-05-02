import numpy as np
from compute_cost import ComputeCost


class GradientDescent:
    def __init__(self):
        pass

    def compute_gradient_descent(self, X, y, theta, alpha, num_iters):
        m = len(y)  # number of samples
        J_history = np.zeros(num_iters)
        compute_cost_obj: ComputeCost = ComputeCost()
        y = y.reshape(m, 1)
        for i in range(num_iters):
            hypothesis = X.transpose().dot(theta)
            theta = theta - (alpha / m) * X.dot(hypothesis - y)
            J_history[i] = compute_cost_obj.compute_cost(X, y, theta)

        return theta
