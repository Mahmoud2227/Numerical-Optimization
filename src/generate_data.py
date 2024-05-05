import numpy as np

def generate_data(num_samples=100, num_features=1, noise=0.1):
    np.random.seed(0)
    """
    Generate random linear data for regression.

    Parameters:
        num_samples (int): Number of data points to generate.
        num_features (int): Number of features (dimensions) in the data.
        noise (float): Amount of noise to add to the data.

    Returns:
        X (ndarray): Array of shape (num_samples, num_features) containing features.
        y (ndarray): Array of shape (num_samples, ) containing target values.
    """
    # Generate random features
    X = np.random.rand(num_samples, num_features)

    # Generate random coefficients for the linear equation
    true_coefficients = np.random.randn(num_features)

    # Generate target values based on the linear equation with added noise
    y = X.dot(true_coefficients) + np.random.normal(scale=noise, size=num_samples)

    return X, y