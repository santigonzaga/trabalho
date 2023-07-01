import numpy as np


def compute_mse(b, w, data):
    squared_errors = ((theta_0 + theta_1 * x - y) ** 2 for x, y in data)
    return sum(squared_errors) / len(data)


def step_gradient(b, w, data, alpha):
    N = len(data)
    b_gradient = 0
    w_gradient = 0

    for i in range(N):
        x = data[i, 0]
        y = data[i, 1]
        error = (b + w * x) - y
        b_gradient += (2/N) * error
        w_gradient += (2/N) * error * x

    new_b = b - alpha * b_gradient
    new_w = w - alpha * w_gradient

    return new_b, new_w

def fit(data, b, w, alpha, num_iterations):
    theta_0_record = [theta_0]
    theta_1_record = [theta_1]

    for _ in range(num_iterations):
        theta_0, theta_1 = step_gradient(theta_0, theta_1, data, alpha)
        theta_0_record.append(theta_0)
        theta_1_record.append(theta_1)

    return theta_0_record, theta_1_record