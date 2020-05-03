import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from feature_nomalize import FeatureNomalize
from gradient_descent import GradientDescent
from warm_up_exercise import WarmUpExercise
from compute_cost import ComputeCost

# --------------- Part 1: Basic function -----------
warm_up_exercise = WarmUpExercise(5)
identity_matrix = warm_up_exercise.create_identity_matrix()

# --------------- Part 2: Plotting -----------
df = pd.read_csv("data/ex1data1.txt")
df = df.to_numpy()
X = df[:, 0]
y = df[:, 1]
m = len(y)

plt.scatter(X, y, c='red')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
# plt.show()

# --------------- Part 3: Cost and Gradient descent -----------

X = np.vstack((np.ones(m), X))  # Add a column of ones to x
theta = np.zeros((2, 1))
iterations = 1500
alpha = 0.01
compute_cost_obj: ComputeCost = ComputeCost()
cost_function_result = compute_cost_obj.compute_cost(X, y, theta)[0]
cost_function_result2 = compute_cost_obj.compute_cost(X, y, np.array([[-1], [2]]))[0]

gradient_descent_obj: GradientDescent = GradientDescent()
new_theta = gradient_descent_obj.compute_gradient_descent(X, y, theta, alpha, iterations)
predict1 = np.array([1, 3.5]).dot(new_theta)[0] * 10000
predict2 = np.array([1, 7]).dot(new_theta)[0] * 10000


# --------------- Part 5: Linear regression with multiple variables -----------
df = pd.read_csv("data/ex1data2.txt")
df = df.to_numpy()
X = df[:, 0:2]
y = df[:, 2]
m = len(y)

X = np.hstack((np.ones(m).reshape([47,1]), X))  # Add a column of ones to x
X = X.transpose()

feature_nomalize: FeatureNomalize = FeatureNomalize()
X = feature_nomalize.do_feature_normalization(X)

alpha = 0.01
num_iters = 400
theta = np.zeros((3, 1))

gradient_descent_obj2: GradientDescent = GradientDescent()
new_theta = gradient_descent_obj2.compute_gradient_descent(X, y, theta, alpha, num_iters)



x = 0
