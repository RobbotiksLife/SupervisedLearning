import numpy as np

def sign(x):
    if x < 0:
        return -1
    elif x == 0:
        return 0
    else:
        return 1

def subgradient_descent(f, initial_x, learning_rate, num_iterations):
    x = initial_x
    for i in range(num_iterations):
        # Compute subgradient
        subgrad = sign(x)
        # Update x
        x = x - learning_rate * subgrad
        # Print current iteration and x value
        print("Iteration:", i+1, "x:", x)
    return x

# Define the function f(x) = |x|
def f(x):
    return abs(x)

# Parameters
initial_x = 5  # Initial guess for x
learning_rate = 0.1  # Learning rate
num_iterations = 100  # Number of iterations

# Apply subgradient descent
optimal_x = subgradient_descent(f, initial_x, learning_rate, num_iterations)
print("Optimal x:", optimal_x)
