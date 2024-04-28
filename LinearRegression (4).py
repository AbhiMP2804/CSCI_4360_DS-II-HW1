import numpy as np

# Setting a seed for reproducibility
np.random.seed(0)

# Creating a toy dataset
# 100 samples and 4 features
X = np.random.randn(100, 4)

# Creating a random weight vector for our linear model (true coefficients)
true_weights = np.random.randn(4)

# Creating responses with some noise
y = X.dot(true_weights) + 1.5 +  np.random.randn(100) * 0.05

# Data augmentation: We append "1" to each of the sample.
X = np.hstack([X, np.ones([100,1])])

# This are the true weights you try to get by running your program. The "1.5"
# is actually the bias term.
true_weights = np.hstack([true_weights, 1.5])

print(true_weights)
print(X.shape, y.shape)


def mean_squared_error(y_true, y_pred):
    """
    Calculate the mean squared error between true and predicted values.
    """
    return np.mean((y_true - y_pred) ** 2)

def sgd_linear_regression(X, y, learning_rate=0.01, n_epochs=300, batch_size=1):
    """
    Stochastic Gradient Descent for a simple linear regression model.
    
    :param X: Input features.
    :param y: Target values.
    :param learning_rate: Learning rate for the SGD.
    :param n_epochs: Number of passes over the dataset.
    :param batch_size: Size of the batch used in each iteration.
    :return: The weights and bias of the trained model.
    """
    n_samples, n_features = X.shape

    # Initialize weights
    weights = np.zeros(n_features)

    # SGD Algorithm
    for epoch in range(n_epochs):
        # Shuffle the dataset
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, n_samples, batch_size):
            X_i = X_shuffled[i:i+batch_size]
            y_i = y_shuffled[i:i+batch_size]

            # Compute predictions
            y_pred = X_i.dot(weights)

            # Compute gradients (Please complete this part)

            # Update weights (Please complete this part)

    return weights

# Train the model using SGD
weights = sgd_linear_regression(X, y)
weights, mean_squared_error(y, X.dot(weights))
