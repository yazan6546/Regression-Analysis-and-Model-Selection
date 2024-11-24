import numpy as np

def linear_regression_fit(X, y):
    # Add a column of ones to X to represent the intercept (bias)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add a column of ones for intercept

    # Calculate the weights using the Normal Equation: w = (X^T * X)^(-1) * X^T * y
    w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    return w

def linear_regression_predict(X, w):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add a column of ones for intercept
    return np.dot(X_b, w)


def gradient_descent(X_train, y, learning_rate=0.01, num_iterations=1000):
   
    errors = []
    W = np.random.randn(X_train.shape[1],) * 0.01  # Initialize W randomly
    b = np.random.randn(1) * 0.01  # Initialize b randomly

    for i in range(num_iterations):
        # TODO: compute errors
        y_pred = np.dot(X_train, W) + b
        difference = y - y_pred

        # TODO: compute steps
        W = W + learning_rate * np.dot(X_train.T, difference) / X_train.size
        b = b + learning_rate * np.sum(difference) / X_train.size

        errors.append(np.mean(difference**2))
        if i % 100 == 0:
            print(f"Iteration {i}: Error = {errors[-1]}")

    W = np.concatenate((b.reshape(-1,), W), axis=0)
    return W, errors
    

    